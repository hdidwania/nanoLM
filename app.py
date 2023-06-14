import argparse

import gradio as gr

import tokenizer as tokenizer_
from inference import TextGenerator

tokenizer_class_dict = {
    "word": tokenizer_.WordTokenizer,
    "char": tokenizer_.CharTokenizer,
}

HEAD_HTML = "<h1> NanoLM - Wikipedia Movie Plots"


def main(args):
    text_generator = TextGenerator(
        tokenizer_type=args.tokenizer,
        tokenizer_path=args.tokenizer_path,
        tokenizer_maxlen=args.maxlen,
        tokenizer_minfreq=args.minfreq,
        model_dims=args.dims,
        model_heads=args.heads,
        model_blocks=args.nblocks,
        model_path=args.model_path,
        sentence_maxlen=args.maxlen,
    )

    with gr.Blocks(title="NanoLM") as demo:
        gr.HTML(value=HEAD_HTML)
        in_context = gr.Text(label="Context")
        with gr.Row():
            in_mode = gr.Radio(
                label="Decoding Mode", choices=["greedy", "beam-search", "sample"]
            )
            in_beams = gr.Slider(
                1, 10, value=1, label="Beam Width", step=1, visible=False
            )
            in_temp = gr.Slider(
                0, 10, value=1.0, label="Sampling Temperature", step=0.5, visible=False
            )
        with gr.Row():
            btn_generate = gr.Button("Generate")

        with gr.Row():
            out_text = gr.Textbox(label="Generated Text")

        def ui_update_on_mode_selection(mode):
            if mode == "greedy":
                return {
                    in_beams: gr.update(visible=False),
                    in_temp: gr.update(visible=False),
                }
            if mode == "beam-search":
                return {
                    in_beams: gr.update(visible=True),
                    in_temp: gr.update(visible=False),
                }
            if mode == "sample":
                return {
                    in_beams: gr.update(visible=False),
                    in_temp: gr.update(visible=True),
                }
            else:
                return {
                    in_beams: gr.update(visible=False),
                    in_temp: gr.update(visible=False),
                }

        in_mode.change(
            fn=ui_update_on_mode_selection,
            inputs=[in_mode],
            outputs=[in_beams, in_temp],
        )
        btn_generate.click(
            fn=text_generator.generate,
            inputs=[in_context, in_mode, in_beams, in_temp],
            outputs=out_text,
        )

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--nblocks", type=int, default=3)
    parser.add_argument(
        "--tokenizer", help="Type of tokenizer", choices=["word", "char"], required=True
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Path of pickle file for loading the tokenizer",
        required=True,
    )
    parser.add_argument(
        "--maxlen", help="Maximum length of sentence", required=True, type=int
    )
    parser.add_argument(
        "--minfreq", help="Minimum freq of words to retain", default=0, type=int
    )
    parser.add_argument("--model-path", help="Path to saved model", required=True)

    args = parser.parse_args()
    main(args)
