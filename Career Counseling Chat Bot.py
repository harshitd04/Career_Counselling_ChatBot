import gradio as gr
from groq import Groq

# --- Setup: Groq API client ---
groq_api = Groq(api_key="")

# --- Response Generator ---
def generate_response(user_input, history):
    if user_input.strip() == "":
        return "\n\n".join(
            [f"*{'You' if m['role'] == 'user' else 'Career Counselor'}:* {m['content']}" for m in history]
        ), history

    # Add system prompt only at beginning
    if not history:
        history.append({
            "role": "system",
            "content": (
                "You are an expert career counselor with experience helping students and professionals. "
                "Provide thoughtful, specific, and personalized advice. Ask questions when needed. "
                "Focus on strengths, interests, and practical next steps."
            )
        })

    history.append({"role": "user", "content": user_input})

    response = groq_api.chat.completions.create(
        model="llama3-70b-8192",
        messages=history,
        temperature=0.7,
        max_tokens=350,
        top_p=1
    )

    # Extract the AI's message and append it to the conversation history
    ai_reply = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": ai_reply})

    # Format the conversation for display
    chat_display = "\n\n".join(
        [f"*{'You' if msg['role'] == 'user' else 'Career Counselor'}:* {msg['content']}" for msg in history if msg['role'] != 'system']
    )

    return chat_display, history

# --- Career Assessment Generator ---
def generate_assessment():
    prompt = "Create 5 self-assessment questions to help someone identify career interests and strengths."
    response = groq_api.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

# --- Career Resources Generator ---
def generate_resources():
    prompt = "List 5 career development resources including resume tools, job search platforms, and skill assessments."
    response = groq_api.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

# --- Career Trends Generator ---
def generate_trends():
    prompt = "What are the top 5 emerging career fields and trends in 2025 job markets?"
    response = groq_api.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

# --- Gradio Interface ---
def interface():
    with gr.Blocks(css="style.css") as demo:
        gr.Markdown("""
        <div class="main-header">
            <h1>🚀 CareerCompass</h1>
            <p>Your AI-powered career counseling assistant</p>
        </div>
        """)

        state = gr.State([])

        with gr.Tabs():
            with gr.Tab("💬 Career Chat"):
                with gr.Column():
                    chatbox = gr.Textbox(
                        label="Conversation",
                        placeholder="Chat appears here...",
                        lines=18,
                        interactive=False,
                        show_copy_button=True
                    )
                    user_input = gr.Textbox(
                        label="Ask your question",
                        placeholder="What guidance do you need?",
                        lines=2
                    )
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear Chat")

            with gr.Tab("🧰 Career Tools"):
                with gr.Column():
                    assess_btn = gr.Button("📊 Generate Self-Assessment Questions")
                    assess_output = gr.Textbox(label="Assessment Questions", lines=6, interactive=False)

                    resource_btn = gr.Button("📚 Show Career Resources")
                    resource_output = gr.Textbox(label="Useful Resources", lines=6, interactive=False)

                    trends_btn = gr.Button("📈 Explore Industry Trends")
                    trends_output = gr.Textbox(label="Industry Trends", lines=6, interactive=False)

        send_btn.click(fn=generate_response, inputs=[user_input, state], outputs=[chatbox, state])
        user_input.submit(fn=generate_response, inputs=[user_input, state], outputs=[chatbox, state])
        clear_btn.click(fn=lambda: ("", []), inputs=None, outputs=[chatbox, state])

        assess_btn.click(fn=generate_assessment, inputs=None, outputs=assess_output)
        resource_btn.click(fn=generate_resources, inputs=None, outputs=resource_output)
        trends_btn.click(fn=generate_trends, inputs=None, outputs=trends_output)

    demo.launch()

# --- Run App ---
if __name__ == "__main__":
    interface()
