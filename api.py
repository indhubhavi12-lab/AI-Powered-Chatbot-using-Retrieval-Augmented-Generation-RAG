response = client.chat.completions.create(
    model=model,
    messages=st.session_state.messages
)