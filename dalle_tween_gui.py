"""Simple GUI for tweening between three keyframes.

Run with:
    python dalle_tween_gui.py

Select start, middle and end images and the app will invoke the Wan2.1 FLF2V
model to generate short video transitions between them. The two videos are
stitched into a single MP4 saved in ``./tween_output``.
"""

import os
import shutil
import logging

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

logger = logging.getLogger(__name__)


from dalle_tween import (
    generate_flf2v_tween,
    combine_videos,
)


class TweenApp:
    """Tkinter application for creating short tween videos."""

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("FLF2V Tween Video Generator")

        self.start_path = tk.StringVar()
        self.middle_path = tk.StringVar()
        self.end_path = tk.StringVar()
        self.count_var = tk.IntVar(value=2)

        tk.Button(master, text="Start Image", command=self.pick_start).grid(row=0, column=0, sticky="e")
        tk.Label(master, textvariable=self.start_path, width=40, anchor="w").grid(row=0, column=1, sticky="w")

        tk.Button(master, text="Middle Image", command=self.pick_middle).grid(row=1, column=0, sticky="e")
        tk.Label(master, textvariable=self.middle_path, width=40, anchor="w").grid(row=1, column=1, sticky="w")

        tk.Button(master, text="End Image", command=self.pick_end).grid(row=2, column=0, sticky="e")
        tk.Label(master, textvariable=self.end_path, width=40, anchor="w").grid(row=2, column=1, sticky="w")

        tk.Label(master, text="Tweens per transition:").grid(row=3, column=0, sticky="e")
        tk.Spinbox(master, from_=1, to=10, textvariable=self.count_var, width=5).grid(row=3, column=1, sticky="w")

        tk.Button(master, text="Generate", command=self.generate).grid(row=4, column=0, columnspan=2)

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
        logger.info(message)

        print(color + message + Colors.RESET)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)

    def generate(self) -> None:
        start = self.start_path.get()
        middle = self.middle_path.get()
        end = self.end_path.get()
        _ = self.count_var.get()

        if not start or not middle or not end:
            messagebox.showerror("Missing Information", "Please select all images.")
            return

        wan2_dir = os.getenv("WAN2_DIR")
        flf2v_model_dir = os.getenv("FLF2V_MODEL_DIR")
        if not wan2_dir or not flf2v_model_dir:
            messagebox.showerror(
                "Configuration Missing",
                "WAN2_DIR and FLF2V_MODEL_DIR environment variables must be set.",
            )
            return

        out_dir = os.path.join(os.getcwd(), "tween_output")
        frames_dir = os.path.join(out_dir, "frames")
        videos_dir = os.path.join(out_dir, "videos")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(videos_dir, exist_ok=True)

        start_frame = os.path.join(frames_dir, "start.png")
        middle_frame = os.path.join(frames_dir, "middle.png")
        end_frame = os.path.join(frames_dir, "end.png")

        shutil.copy(start, start_frame)
        shutil.copy(middle, middle_frame)
        shutil.copy(end, end_frame)

        try:
            self.log_message("Generating video start->middle...")
            tween1 = os.path.join(videos_dir, "tween_01.mp4")
            generate_flf2v_tween(start_frame, middle_frame, tween1, wan2_dir, flf2v_model_dir)

            self.log_message("Generating video middle->end...")
            tween2 = os.path.join(videos_dir, "tween_02.mp4")
            generate_flf2v_tween(middle_frame, end_frame, tween2, wan2_dir, flf2v_model_dir)

            final_video = os.path.join(out_dir, "tween.mp4")
            combine_videos([tween1, tween2], final_video)
            self.log_message(f"Video saved to {final_video}", color=Colors.GREEN)
        except Exception as exc:
            self.log_message(f"Error: {exc}", color=Colors.RED)
            messagebox.showerror("Generation Failed", str(exc))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    root = tk.Tk()
    app = TweenApp(root)
    root.mainloop()
