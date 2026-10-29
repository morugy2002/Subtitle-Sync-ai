"""SubtitleSync AI - رابط گرافیکی"""
import os
import queue
import sys
import threading
import traceback
import webbrowser

if sys.stdout is None:  # حالت --windowed در PyInstaller
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import core

ENCODINGS = ["auto", "utf-8", "cp1256", "cp1251", "cp1252", "cp1253", "iso-8859-9", "gbk", "big5", "shift_jis", "euc-kr"]
MEDIA_TYPES = [("Video / audio", "*.mkv *.mp4 *.avi *.mov *.webm *.m4v *.ts *.mp3 *.wav *.m4a *.flac *.ogg"), ("All files", "*.*")]
SUB_TYPES = [("Subtitles", "*.srt *.vtt *.ass *.ssa"), ("All files", "*.*")]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SubtitleSync AI")
        self.geometry("820x760")
        self.minsize(720, 640)
        self.q = queue.Queue()
        self.job = None
        self.busy = False
        self.buttons = []

        # --- variables
        self.fa, self.ref, self.out = (tk.StringVar() for _ in range(3))
        self.mode = tk.StringVar(value="srt")
        self.model = tk.StringVar(value="minilm")
        self.sim = tk.DoubleVar(value=core.MODELS["minilm"]["min_sim"])
        self.enc = tk.StringVar(value="auto")
        self.w_model = tk.StringVar()
        self.lang = tk.StringVar(value="")
        self.fixfa = tk.BooleanVar(value=True)
        self.rle = tk.BooleanVar(value=False)
        self.bili = tk.BooleanVar(value=False)
        self.bom = tk.BooleanVar(value=True)
        self.g_media, self.g_out = tk.StringVar(), tk.StringVar()
        self.g_model = tk.StringVar()
        self.g_lang = tk.StringVar(value="")
        self.g_tr = tk.BooleanVar(value=False)
        self.c_wmodel = tk.StringVar(value="small-q5_1")

        nb = ttk.Notebook(self)
        nb.pack(fill="x", padx=10, pady=(10, 4))
        self.tab_sync, self.tab_gen, self.tab_comp = ttk.Frame(nb, padding=10), ttk.Frame(nb, padding=10), ttk.Frame(nb, padding=10)
        nb.add(self.tab_sync, text="  Sync subtitle  ")
        nb.add(self.tab_gen, text="  Create subtitle from video  ")
        nb.add(self.tab_comp, text="  Components  ")
        self.build_sync()
        self.build_gen()
        self.build_comp()

        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.pack(fill="both", expand=True)
        bar_row = ttk.Frame(bottom)
        bar_row.pack(fill="x")
        self.bar = ttk.Progressbar(bar_row, mode="determinate", maximum=100)
        self.bar.pack(side="left", fill="x", expand=True)
        self.cancel_btn = ttk.Button(bar_row, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=(8, 0))
        self.log_box = scrolledtext.ScrolledText(bottom, height=12, state="disabled", font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True, pady=(6, 0))
        self.refresh()
        self.after(100, self.poll)

    # ---------------------------------------------------------- helpers
    def row(self, parent, label, var, cmd, r, width=22):
        ttk.Label(parent, text=label, width=width).grid(row=r, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=var).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(parent, text="Browse...", command=cmd).grid(row=r, column=2)
        parent.columnconfigure(1, weight=1)

    def pick(self, var, types, save=False, ext=".srt"):
        f = (filedialog.asksaveasfilename(defaultextension=ext, filetypes=types) if save
             else filedialog.askopenfilename(filetypes=types))
        if f:
            var.set(f)
        return f

    def add_button(self, parent, **kw):
        b = ttk.Button(parent, **kw)
        self.buttons.append(b)
        return b

    # -------------------------------------------------------- tab: sync
    def build_sync(self):
        t = self.tab_sync
        self.row(t, "Subtitle to fix:", self.fa, self.pick_fa, 0)
        box = ttk.LabelFrame(t, text="Timing reference", padding=8)
        box.grid(row=1, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Radiobutton(box, text="A subtitle that is already in sync (any language)", value="srt",
                        variable=self.mode).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(box, text="The video / audio file (speech recognition, needs Whisper)", value="media",
                        variable=self.mode).grid(row=1, column=0, columnspan=3, sticky="w")
        self.row(box, "Reference file:", self.ref, lambda: self.pick(
            self.ref, SUB_TYPES if self.mode.get() == "srt" else MEDIA_TYPES), 2)
        self.row(t, "Save result as:", self.out, lambda: self.pick(self.out, [("SubRip", "*.srt")], True), 2)

        o = ttk.LabelFrame(t, text="Options", padding=8)
        o.grid(row=3, column=0, columnspan=3, sticky="ew", pady=6)
        ttk.Label(o, text="Matching model:").grid(row=0, column=0, sticky="w")
        self.model_cb = ttk.Combobox(o, textvariable=self.model, state="readonly", width=44)
        self.model_cb.grid(row=0, column=1, sticky="w", padx=6)
        self.model_cb.bind("<<ComboboxSelected>>", self.on_model)
        ttk.Label(o, text="Min similarity:").grid(row=0, column=2, sticky="e", padx=(12, 0))
        ttk.Spinbox(o, from_=0.2, to=0.9, increment=0.05, textvariable=self.sim, width=6).grid(row=0, column=3, padx=6)
        ttk.Label(o, text="Whisper model:").grid(row=1, column=0, sticky="w", pady=3)
        self.w_cb = ttk.Combobox(o, textvariable=self.w_model, state="readonly", width=44)
        self.w_cb.grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(o, text="Speech language:").grid(row=1, column=2, sticky="e", padx=(12, 0))
        ttk.Entry(o, textvariable=self.lang, width=6).grid(row=1, column=3, padx=6)
        ttk.Label(o, text="File encoding:").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Combobox(o, textvariable=self.enc, values=ENCODINGS, width=12).grid(row=2, column=1, sticky="w", padx=6)
        c = ttk.Frame(o)
        c.grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Checkbutton(c, text="Fix Persian text (Arabic ي/ك, digits)", variable=self.fixfa).pack(anchor="w")
        ttk.Checkbutton(c, text="Bilingual output (reference text under each line)", variable=self.bili).pack(anchor="w")
        ttk.Checkbutton(c, text="Fix RTL punctuation (adds invisible marks; try if dots jump to the wrong side)",
                        variable=self.rle).pack(anchor="w")
        ttk.Checkbutton(c, text="UTF-8 with BOM (recommended for Persian)", variable=self.bom).pack(anchor="w")
        ttk.Label(o, text="Speech language e.g. en (empty = auto). Whisper options apply to video mode only.",
                  foreground="#666").grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.add_button(t, text="Synchronize", command=self.start_sync).grid(row=4, column=0, columnspan=3, pady=8)

    def pick_fa(self):
        f = self.pick(self.fa, SUB_TYPES)
        if f:
            self.out.set(os.path.splitext(f)[0] + ".synced.srt")

    def on_model(self, _=None):
        key = self.model.get().split(" - ")[0]
        self.model.set(key)
        if key in core.MODELS:
            self.sim.set(core.MODELS[key]["min_sim"])

    # -------------------------------------------------------- tab: generate
    def build_gen(self):
        t = self.tab_gen
        self.row(t, "Video / audio file:", self.g_media, lambda: self.pick_gen(), 0)
        self.row(t, "Save subtitle as:", self.g_out, lambda: self.pick(self.g_out, [("SubRip", "*.srt")], True), 1)
        o = ttk.LabelFrame(t, text="Options", padding=8)
        o.grid(row=2, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(o, text="Whisper model:").grid(row=0, column=0, sticky="w")
        self.g_cb = ttk.Combobox(o, textvariable=self.g_model, state="readonly", width=40)
        self.g_cb.grid(row=0, column=1, padx=6, sticky="w")
        ttk.Label(o, text="Speech language:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(o, textvariable=self.g_lang, width=8).grid(row=1, column=1, padx=6, sticky="w")
        ttk.Checkbutton(o, text="Translate speech to English", variable=self.g_tr).grid(
            row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(o, text="Language: e.g. en, fa, de, fr (empty = auto-detect). Translation works to English only.",
                  foreground="#666").grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.add_button(t, text="Create subtitle", command=self.start_gen).grid(row=3, column=0, columnspan=3, pady=8)

    def pick_gen(self):
        f = self.pick(self.g_media, MEDIA_TYPES)
        if f:
            self.g_out.set(os.path.splitext(f)[0] + ".srt")

    # -------------------------------------------------------- tab: components
    def build_comp(self):
        t = self.tab_comp
        t.columnconfigure(1, weight=1)
        ttk.Label(t, text="Needed for syncing two subtitles:", font=("", 9, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w")
        self.st = {}
        r = 1
        for key, m in core.MODELS.items():
            ttk.Label(t, text=f"{m['title']}  (~{m['approx_mb']} MB)", wraplength=420, justify="left").grid(
                row=r, column=0, sticky="w", pady=3)
            self.st[key] = ttk.Label(t, text="")
            self.st[key].grid(row=r, column=1, sticky="w", padx=8)
            self.add_button(t, text="Download", command=lambda k=key: self.start_download_model(k)).grid(row=r, column=2)
            r += 1
        ttk.Separator(t).grid(row=r, column=0, columnspan=4, sticky="ew", pady=8)
        r += 1
        ttk.Label(t, text="Optional - speech recognition (Whisper):", font=("", 9, "bold")).grid(
            row=r, column=0, columnspan=4, sticky="w")
        r += 1
        ttk.Label(t, text="Whisper model:").grid(row=r, column=0, sticky="w", pady=3)
        self.cw = ttk.Combobox(t, textvariable=self.c_wmodel, state="readonly", width=30,
                               values=[f"{k}  (~{v[1]} MB)" for k, v in core.WHISPER_MODELS.items()])
        self.cw.grid(row=r, column=1, sticky="w", padx=8)
        self.cw.current(2)
        self.add_button(t, text="Download", command=self.start_download_whisper).grid(row=r, column=2)
        r += 1
        self.cw_status = ttk.Label(t, text="")
        self.cw_status.grid(row=r, column=1, sticky="w", padx=8)
        r += 1
        for key, tool in core.TOOLS.items():
            ttk.Label(t, text=tool["title"]).grid(row=r, column=0, sticky="w", pady=6)
            self.st[key] = ttk.Label(t, text="", wraplength=300)
            self.st[key].grid(row=r, column=1, sticky="w", padx=8)
            f = ttk.Frame(t)
            f.grid(row=r, column=2, columnspan=2, sticky="e")
            ttk.Button(f, text="Download page", command=lambda p=tool["page"]: webbrowser.open(p)).pack(side="left")
            ttk.Button(f, text="Install from ZIP...", command=lambda k=key: self.install_zip(k)).pack(side="left", padx=3)
            ttk.Button(f, text="Locate...", command=lambda k=key: self.locate(k)).pack(side="left")
            r += 1
        hint = ("whisper.cpp: from the releases page download whisper-bin-x64.zip (CPU) or the whisper-cublas-...-x64.zip "
                "(NVIDIA GPU, faster), then use 'Install from ZIP'.\n"
                "ffmpeg: download a Windows build (e.g. 'release essentials' ZIP), then 'Install from ZIP'. "
                "Or install it with: winget install ffmpeg")
        ttk.Label(t, text=hint, foreground="#555", wraplength=740, justify="left").grid(
            row=r, column=0, columnspan=4, sticky="w", pady=(8, 0))

    def wkey(self, text):
        return text.split("  ")[0].strip()

    def install_zip(self, key):
        f = filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])
        if f:
            self.run_task(lambda job, log, progress: core.install_tool_zip(key, f), f"Installing {key} ...")

    def locate(self, key):
        f = filedialog.askopenfilename(filetypes=[("Program", "*.exe"), ("All", "*.*")])
        if f:
            core.set_tool_path(key, f)
            self.refresh()

    def refresh(self):
        inst = [k for k in core.MODELS if core.model_installed(k)]
        self.model_cb["values"] = list(core.MODELS)
        if self.model.get() not in core.MODELS:
            self.model.set("minilm")
        wi = [k for k in core.WHISPER_MODELS if core.whisper_model_installed(k)]
        for cb, var in ((self.w_cb, self.w_model), (self.g_cb, self.g_model)):
            cb["values"] = wi
            if var.get() not in wi:
                var.set(wi[0] if wi else "")
        for k in core.MODELS:
            self.st[k].config(text="Installed" if k in inst else "Not downloaded",
                              foreground="#0a7a2f" if k in inst else "#b00020")
        for k in core.TOOLS:
            p = core.find_tool(k)
            self.st[k].config(text=p if p else "Not found", foreground="#0a7a2f" if p else "#b00020")
        self.cw_status.config(text="Installed models: " + (", ".join(wi) if wi else "none"))

    # ---------------------------------------------------------- tasks
    def log(self, s):
        self.q.put(("log", s))

    def progress(self, done, total):
        self.q.put(("prog", done, total))

    def set_busy(self, busy):
        self.busy = busy
        for b in self.buttons:
            b.config(state="disabled" if busy else "normal")
        self.cancel_btn.config(state="normal" if busy else "disabled")
        if not busy:
            self.bar["value"] = 0

    def run_task(self, fn, label="Working ..."):
        if self.busy:
            return
        self.set_busy(True)
        self.job = core.Job()
        self.log(label)

        def work():
            try:
                fn(self.job, self.log, self.progress)
                self.q.put(("done", None))
            except core.Cancelled:
                self.q.put(("done", "Cancelled."))
            except Exception as e:  # noqa: BLE001
                self.q.put(("log", traceback.format_exc()))
                self.q.put(("done", f"{e}"))

        threading.Thread(target=work, daemon=True).start()

    def cancel(self):
        if self.job:
            self.job.cancel()

    def poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "log":
                    self.log_box.config(state="normal")
                    self.log_box.insert("end", m[1] + "\n")
                    self.log_box.see("end")
                    self.log_box.config(state="disabled")
                elif m[0] == "prog":
                    self.bar["value"] = 100 * m[1] / m[2] if m[2] else 0
                elif m[0] == "done":
                    self.set_busy(False)
                    self.refresh()
                    if m[1] == "Cancelled.":
                        self.log("Cancelled.")
                    elif m[1]:
                        messagebox.showerror("Error", m[1])
                    else:
                        self.log("Done.")
        except queue.Empty:
            pass
        self.after(100, self.poll)

    # ------------------------------------------------------ task starters
    def start_download_model(self, key):
        self.run_task(lambda job, log, progress: core.download_model(key, progress, job, log),
                      f"Downloading {core.MODELS[key]['title']} ...")

    def start_download_whisper(self):
        key = self.wkey(self.cw.get())
        self.run_task(lambda job, log, progress: core.download_whisper_model(key, progress, job, log),
                      f"Downloading Whisper model {key} ...")

    def start_sync(self):
        if not (self.fa.get() and self.ref.get() and self.out.get()):
            messagebox.showwarning("Missing input", "Choose the subtitle, the reference and the output file.")
            return
        media = self.mode.get() == "media"
        kw = dict(model_key=self.model.get(), min_sim=float(self.sim.get()),
                  encoding=None if self.enc.get() == "auto" else self.enc.get(),
                  fix_fa=self.fixfa.get(), rle=self.rle.get(), bilingual=self.bili.get(), bom=self.bom.get(),
                  whisper_key=self.w_model.get() or None, language=self.lang.get().strip() or None)
        fa, ref, out = self.fa.get(), self.ref.get(), self.out.get()

        def fn(job, log, progress):
            info = core.sync_pipeline(fa, out, None if media else ref, ref if media else None,
                                      job=job, log=log, progress=progress, **kw)
            self.q.put(("log", f"Result: {info['anchors']}/{info['total']} lines matched ({info['coverage']:.0%})."))

        self.run_task(fn, "Starting ...")

    def start_gen(self):
        if not (self.g_media.get() and self.g_out.get()):
            messagebox.showwarning("Missing input", "Choose the video and the output file.")
            return
        m, o = self.g_media.get(), self.g_out.get()
        key, lang, tr = self.g_model.get() or None, self.g_lang.get().strip() or None, self.g_tr.get()
        self.run_task(lambda job, log, progress: core.generate_subtitle(
            m, o, key, lang, tr, True, job, log, progress), "Starting ...")


if __name__ == "__main__":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    App().mainloop()
