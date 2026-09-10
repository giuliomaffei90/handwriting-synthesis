#!/usr/bin/env python3
"""Handwriting - a minimal desktop front end for handwriting synthesis.

Type text, pick a style, generate, save as PNG or SVG. No terminal, no
TensorFlow: hw.engine runs the original pretrained model on numpy alone.
"""
import os
import queue
import random
import sys
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hw import drawing, render  # noqa: E402
from hw.engine import (STYLES_DIR, WEIGHTS_PATH, Model, available_styles,
                       style_strokes)  # noqa: E402

PAPER = '#ffffff'
CHECKER = '#eeeeec'
DEFAULT_TEXT = "Everything is going to be alright."
DEFAULT_BIAS = 1.0


class Cancelled(Exception):
    pass


class App(ttk.Frame):

    def __init__(self, root):
        ttk.Frame.__init__(self, root, padding=12)
        self.root = root
        self.model = None
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        self.polylines, self.page_size = [], (1.0, 1.0)
        self.color = '#111111'
        self.photos = {}          # canvas -> PhotoImage, or Tk drops the image
        self.redraw_job = None
        self.styles = available_styles()

        self.grid(sticky='nsew')
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self._build()
        self._draw_style_preview()
        self.after(50, self._poll)

    # ------------------------------------------------------------------ ui

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        self.text = tk.Text(self, height=5, wrap='word', undo=True,
                            font=('Helvetica', 13), highlightthickness=1,
                            relief='solid', borderwidth=1)
        self.text.insert('1.0', DEFAULT_TEXT)
        self.text.grid(row=0, column=0, sticky='ew')

        self.notice = ttk.Label(self, text='', foreground='#b45309')
        self.notice.grid(row=1, column=0, sticky='w', pady=(4, 8))

        controls = ttk.Frame(self)
        controls.grid(row=2, column=0, sticky='ew')
        controls.columnconfigure(6, weight=1)

        ttk.Label(controls, text='Style').grid(row=0, column=0, padx=(0, 6))
        self.style_var = tk.StringVar(value=str(self.styles[9 % len(self.styles)]))
        style_box = ttk.Combobox(controls, textvariable=self.style_var, width=5,
                                 state='readonly', values=[str(s) for s in self.styles])
        style_box.grid(row=0, column=1)
        style_box.bind('<<ComboboxSelected>>', lambda _e: self._draw_style_preview())

        self.bias = tk.DoubleVar(value=DEFAULT_BIAS)
        self.bias_label = ttk.Label(controls, text='Neatness {:.2f}'.format(DEFAULT_BIAS),
                                    width=15)
        self.bias_label.grid(row=0, column=2, padx=(18, 6))
        ttk.Scale(controls, from_=0.3, to=2.0, variable=self.bias, length=110,
                  orient='horizontal', command=self._on_bias).grid(row=0, column=3)

        ttk.Label(controls, text='Pen').grid(row=0, column=4, padx=(18, 6))
        self.width = tk.DoubleVar(value=2.0)
        ttk.Scale(controls, from_=0.7, to=6.0, variable=self.width, length=90,
                  orient='horizontal',
                  command=lambda _v: self._schedule_redraw()).grid(row=0, column=5)

        self.color_button = tk.Canvas(controls, width=34, height=18, background=self.color,
                                      highlightthickness=1, highlightbackground='#8c8c8c',
                                      cursor='hand2')
        self.color_button.grid(row=0, column=6, padx=(12, 0), sticky='w')
        self.color_button.bind('<Button-1>', lambda _e: self._pick_color())

        self.hint = ttk.Label(
            controls, foreground='#8a8a8a',
            text='Neatness steadies the hand: low wanders and scrawls, high writes '
                 'carefully. Images are saved with a transparent background.')
        self.hint.grid(row=1, column=0, columnspan=7, sticky='w', pady=(8, 0))

        self.style_preview = tk.Canvas(self, height=52, background=PAPER,
                                       highlightthickness=1, highlightbackground='#d4d4d4')
        self.style_preview.grid(row=3, column=0, sticky='ew', pady=(10, 10))
        self.style_preview.bind('<Configure>', lambda _e: self._draw_style_preview())

        self.canvas = tk.Canvas(self, background=PAPER, highlightthickness=1,
                                highlightbackground='#d4d4d4', height=260)
        self.canvas.grid(row=4, column=0, sticky='nsew')
        self.canvas.bind('<Configure>', lambda _e: self._schedule_redraw())

        bottom = ttk.Frame(self)
        bottom.grid(row=5, column=0, sticky='ew', pady=(10, 0))
        bottom.columnconfigure(2, weight=1)

        self.write_button = ttk.Button(bottom, text='Write', command=self._on_write)
        self.write_button.grid(row=0, column=0)
        self.root.bind('<Command-Return>', lambda _e: self._on_write())

        self.progress = ttk.Progressbar(bottom, length=140, mode='determinate', maximum=1.0)
        self.progress.grid(row=0, column=1, padx=10)
        self.status = ttk.Label(bottom, text='', foreground='#525252')
        self.status.grid(row=0, column=2, sticky='w')

        self.png_button = ttk.Button(bottom, text='Save PNG', command=self._save_png,
                                     state='disabled')
        self.png_button.grid(row=0, column=4, padx=(10, 0))
        self.svg_button = ttk.Button(bottom, text='Save SVG', command=self._save_svg,
                                     state='disabled')
        self.svg_button.grid(row=0, column=5, padx=(6, 0))

    def _on_bias(self, _value=None):
        self.bias_label.configure(text='Neatness {:.2f}'.format(self.bias.get()))

    def _pick_color(self):
        chosen = colorchooser.askcolor(color=self.color, parent=self.root)[1]
        if chosen:
            self.color = chosen
            self.color_button.configure(background=chosen)
            self._schedule_redraw()

    # ------------------------------------------------------------- drawing

    def _checkerboard(self, size, box=9):
        """Light chequer, so it is obvious the saved image has no background."""
        image = Image.new('RGBA', size, PAPER)
        draw = ImageDraw.Draw(image)
        for row, y in enumerate(range(0, size[1], box)):
            for x in range((row % 2) * box, size[0], 2 * box):
                draw.rectangle([x, y, x + box - 1, y + box - 1], fill=CHECKER)
        return image

    def _paint(self, canvas, polylines, size, width, color, pad=8, max_zoom=1.0,
               checkerboard=False):
        """Draw through the same renderer that saves the file, so what you see
        on screen is what lands in the PNG."""
        canvas.delete('all')
        self.photos.pop(canvas, None)
        if not polylines:
            return
        room_x = max(canvas.winfo_width() - 2 * pad, 10)
        room_y = max(canvas.winfo_height() - 2 * pad, 10)
        zoom = min(room_x / size[0], room_y / size[1], max_zoom)

        image = render.to_image(polylines, size, stroke_width=width, color=color,
                               dpi_scale=zoom)
        if checkerboard:
            board = self._checkerboard(image.size)
            board.alpha_composite(image)
            image = board

        photo = ImageTk.PhotoImage(image)
        self.photos[canvas] = photo
        canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                            image=photo)

    def _schedule_redraw(self, delay=70):
        if self.redraw_job is not None:
            self.after_cancel(self.redraw_job)
        self.redraw_job = self.after(delay, self._redraw)

    def _redraw(self):
        self.redraw_job = None
        # the renderer holds up when enlarged, so fill the canvas up to 2x
        self._paint(self.canvas, self.polylines, self.page_size,
                    self.width.get(), self.color, max_zoom=2.0, checkerboard=True)

    def _draw_style_preview(self):
        style = int(self.style_var.get())
        polylines, size = render.layout([style_strokes(style)], margin=4)
        self._paint(self.style_preview, polylines, size, 1.4, '#525252', pad=4)

    # ---------------------------------------------------------- generation

    def _on_write(self):
        if self.worker and self.worker.is_alive():
            self.cancel.set()
            return
        raw = self.text.get('1.0', 'end').rstrip('\n')
        clean, dropped = drawing.sanitize(raw)
        lines = drawing.wrap(clean)
        if not any(line.strip() for line in lines):
            messagebox.showinfo('Nothing to write', 'Type some text first.', parent=self.root)
            return
        self.notice.configure(
            text='Skipped characters the model was never taught: ' + ' '.join(dropped)
            if dropped else '')

        self.cancel.clear()
        self.write_button.configure(text='Stop')
        self.png_button.configure(state='disabled')
        self.svg_button.configure(state='disabled')
        self.status.configure(text='Loading model...' if self.model is None else 'Writing...')
        self.progress.configure(value=0.0)

        style = int(self.style_var.get())
        bias = float(self.bias.get())
        seed = random.randrange(2 ** 31)
        self.worker = threading.Thread(target=self._generate, args=(lines, style, bias, seed),
                                       daemon=True)
        self.worker.start()

    def _generate(self, lines, style, bias, seed):
        def progress(fraction):
            if self.cancel.is_set():
                raise Cancelled()
            self.events.put(('progress', fraction))

        try:
            if self.model is None:
                self.model = Model()
            self.events.put(('progress', 0.0))
            strokes = self.model.generate(lines, style=style, bias=bias, seed=seed,
                                          progress=progress)
            self.events.put(('done', strokes))
        except Cancelled:
            self.events.put(('cancelled', None))
        except Exception as error:  # surface model/IO failures in the UI
            self.events.put(('error', '{}: {}'.format(type(error).__name__, error)))

    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == 'progress':
                    self.progress.configure(value=payload)
                elif kind == 'done':
                    self._finish(payload)
                elif kind == 'cancelled':
                    self._reset('Stopped.')
                elif kind == 'error':
                    self._reset('')
                    messagebox.showerror('Generation failed', payload, parent=self.root)
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _finish(self, strokes):
        self.polylines, self.page_size = render.layout(strokes)
        self._redraw()
        points = sum(len(s) for s in self.polylines)
        self._reset('{} strokes, {} points'.format(len(self.polylines), points))
        if self.polylines:
            self.png_button.configure(state='normal')
            self.svg_button.configure(state='normal')

    def _reset(self, message):
        self.progress.configure(value=0.0)
        self.write_button.configure(text='Write')
        self.status.configure(text=message)

    # ---------------------------------------------------------------- save

    def _save_png(self):
        path = filedialog.asksaveasfilename(
            parent=self.root, defaultextension='.png', initialfile='handwriting.png',
            filetypes=[('PNG image', '*.png')])
        if not path:
            return
        image = render.to_image(self.polylines, self.page_size, stroke_width=self.width.get(),
                               color=self.color, dpi_scale=2.0)
        image.save(path)
        self.status.configure(text='Saved ' + os.path.basename(path))

    def _save_svg(self):
        path = filedialog.asksaveasfilename(
            parent=self.root, defaultextension='.svg', initialfile='handwriting.svg',
            filetypes=[('SVG vector', '*.svg')])
        if not path:
            return
        svg = render.to_svg(self.polylines, self.page_size, stroke_width=self.width.get(),
                            color=self.color)
        with open(path, 'w') as handle:
            handle.write(svg)
        self.status.configure(text='Saved ' + os.path.basename(path))


def selftest():
    """Check a built bundle can find its weights and styles: --selftest."""
    model = Model()
    strokes = model.generate(['handwriting self test'], style=available_styles()[0],
                             bias=0.75, seed=0)
    polylines, size = render.layout(strokes)
    print('weights: {}\nstyles:  {}\npoints:  {}\nstrokes: {}\npage:    {:.0f}x{:.0f}'.format(
        WEIGHTS_PATH, STYLES_DIR, sum(len(s) for s in strokes), len(polylines), *size))
    assert polylines, 'no strokes generated'

    # the preview goes through PIL -> Tk, which needs Pillow's _imagingtk to
    # have been bundled: check it here rather than discovering it at runtime
    root = tk.Tk()
    root.withdraw()
    photo = ImageTk.PhotoImage(render.to_image(polylines, size, dpi_scale=0.4))
    print('preview: {}x{} through ImageTk'.format(photo.width(), photo.height()))
    root.destroy()
    print('SELFTEST OK')


def main():
    if '--selftest' in sys.argv:
        return selftest()
    root = tk.Tk()
    root.title('Handwriting')
    root.geometry('820x680')
    root.minsize(640, 520)
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
