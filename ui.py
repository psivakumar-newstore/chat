# ui.py
import streamlit as st
import importlib
import traceback

st.set_page_config(page_title="NewStore Agent", layout="wide")

st.title("🤖 NewStore Agent")
st.caption(
    "Type natural language or a Doc Name/Alias from your CSV. "
    "The agent maps your request to the right method+URL and calls it."
)

# Sidebar: Setup (no auto-build at load)
st.sidebar.header("⚙️ Setup")
tenant = st.sidebar.text_input("Tenant (subdomain):", value="retailsuccess-sandbox")
csv_path = st.sidebar.text_input("CSV file path:", value="url_tagged_with_alias.csv")
headers_json = st.sidebar.text_area(
    "Default headers JSON (optional):",
    value=f'{{"x-tenant":"{tenant}"}}',
)

def _reload_and_build():
    """Reload the agent module and build a fresh agent ONLY when user clicks."""
    try:
        import agent_with_lambda_tool as awlt
        awlt = importlib.reload(awlt)

        # Apply runtime config
        awlt.set_tenant_base(tenant.strip())
        awlt.set_csv_path(csv_path.strip())
        awlt.set_default_headers_json(headers_json.strip())

        with st.spinner("Building agent…"):
            agent_obj = awlt.build_agent()
        return awlt, agent_obj
    except Exception as e:
        st.error("❌ Error while (re)building the agent.")
        st.code("".join(traceback.format_exc()))
        return None, None

colA, _ = st.sidebar.columns([1, 3])
if st.sidebar.button("✅ Apply configuration & (re)load agent", use_container_width=True):
    mod, agent_obj = _reload_and_build()
    if agent_obj:
        st.session_state["agent"] = agent_obj
        st.session_state["mod"] = mod
        st.success("Agent is ready.")

# Input row
user_prompt = st.text_input(
    "Type your request… (e.g., Get Associate App configuration)",
    "",
    placeholder="Try: Get Associate App configuration",
)

c1, c2 = st.columns([1, 1])
ask = c1.button("Ask", type="primary")
reload_btn = c2.button("Reload agent")

if reload_btn:
    st.session_state.pop("agent", None)
    st.session_state.pop("mod", None)
    st.info("Cleared agent. Click 'Apply configuration' to rebuild.")

# Only build on demand; if user presses Ask without an agent, warn.
if ask:
    if "agent" not in st.session_state:
        st.warning("Please click **Apply configuration & (re)load agent** first.")
    elif not user_prompt.strip():
        st.warning("Please type a request.")
    else:
        agent = st.session_state["agent"]
        try:
            with st.spinner("Thinking & calling APIs (with timeouts)…"):
                response = agent.run(user_prompt)
            st.subheader("Response")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Error while running agent: {e}")
            st.code("".join(traceback.format_exc()))

# Diagnostics
st.divider()
st.subheader("🔎 Diagnostics")
if "mod" in st.session_state:
    snap = st.session_state["mod"].debug_snapshot()
    st.json(snap)
else:
    st.info("No agent loaded yet. Configure in the sidebar and click the green button.")