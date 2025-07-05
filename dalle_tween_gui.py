"""Simple GUI for generating tweened GIFs using DALL·E 3.

Run with:
    python dalle_tween_gui.py

The application lets you select start, middle, and end keyframes and a frame
count. It generates DALL·E 3 prompts and images for each transition and combines
the results into an animated GIF.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from typing import List


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"

from dalle_tween import (
    generate_dalle_prompts,
    generate_dalle_images,
    combine_images_to_gif,
)


class TweenApp:
    """Tkinter application for creating GIFs from keyframes."""

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("DALL·E Tween GIF Generator")

        self.api_key_var = tk.StringVar()
        self.start_path = tk.StringVar()
        self.middle_path = tk.StringVar()
        self.end_path = tk.StringVar()
        self.count_var = tk.IntVar(value=2)

        tk.Label(master, text="OpenAI API Key:").grid(row=0, column=0, sticky="e")
        tk.Entry(master, textvariable=self.api_key_var, width=40, show="*").grid(row=0, column=1, sticky="w")

        tk.Button(master, text="Start Image", command=self.pick_start).grid(row=1, column=0, sticky="e")
        tk.Label(master, textvariable=self.start_path, width=40, anchor="w").grid(row=1, column=1, sticky="w")

        tk.Button(master, text="Middle Image", command=self.pick_middle).grid(row=2, column=0, sticky="e")
        tk.Label(master, textvariable=self.middle_path, width=40, anchor="w").grid(row=2, column=1, sticky="w")

        tk.Button(master, text="End Image", command=self.pick_end).grid(row=3, column=0, sticky="e")
        tk.Label(master, textvariable=self.end_path, width=40, anchor="w").grid(row=3, column=1, sticky="w")

        tk.Label(master, text="Tweens per transition:").grid(row=4, column=0, sticky="e")
        tk.Spinbox(master, from_=1, to=10, textvariable=self.count_var, width=5).grid(row=4, column=1, sticky="w")

        tk.Button(master, text="Generate", command=self.generate).grid(row=5, column=0, columnspan=2)

        self.log = scrolledtext.ScrolledText(master, width=60, height=15)
        self.log.grid(row=6, column=0, columnspan=2, pady=10)

    # ------------------------------------------------------------------
    # File selection helpers
    # ------------------------------------------------------------------
    def pick_start(self) -> None:
        path = filedialog.askopenfilename(title="Select start keyframe")
        if path:
            self.start_path.set(path)

    def pick_middle(self) -> None:
        path = filedialog.askopenfilename(title="Select middle keyframe")
        if path:
            self.middle_path.set(path)

    def pick_end(self) -> None:
        path = filedialog.askopenfilename(title="Select end keyframe")
        if path:
            self.end_path.set(path)

    # ------------------------------------------------------------------
    def log_message(self, message: str, color: str = Colors.BLUE) -> None:
        """Print and display a message."""
        print(color + message + Colors.RESET)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)

    def generate(self) -> None:
        api_key = self.api_key_var.get().strip()
        start = self.start_path.get()
        middle = self.middle_path.get()
        end = self.end_path.get()
        count = self.count_var.get()

        if not api_key or not start or not middle or not end:
            messagebox.showerror("Missing Information", "Please select all images and provide the API key.")
            return

        out_dir = os.path.join(os.getcwd(), "tween_output")
        frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        all_images: List[str] = []

        try:
            self.log_message("Generating prompts for start->middle...")
            prompts = generate_dalle_prompts(start, middle, count, api_key)
            self.log_message(f"Generated {len(prompts)} prompts")
            images = generate_dalle_images(prompts, frames_dir, api_key)
            all_images.extend(images)

            self.log_message("Generating prompts for middle->end...")
            prompts = generate_dalle_prompts(middle, end, count, api_key)
            self.log_message(f"Generated {len(prompts)} prompts")
            images = generate_dalle_images(prompts, frames_dir, api_key)
            all_images.extend(images)

            gif_path = os.path.join(out_dir, "tween.gif")
            combine_images_to_gif(all_images, gif_path)
            self.log_message(f"GIF saved to {gif_path}", color=Colors.GREEN)
        except Exception as exc:
            self.log_message(f"Error: {exc}", color=Colors.RED)
            messagebox.showerror("Generation Failed", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    app = TweenApp(root)
    root.mainloop()
