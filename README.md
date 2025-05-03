# 3D Scan STL Origin Aligner

A small Python tool to help **align STL files** (e.g. from 3D scanners) to the **origin and axes** for cleaner integration in CAD or modeling workflows.

I built this out of frustration: I own two Creality 3D scanners, and none of the “affordable” software options support proper mesh alignment. Sure, Fusion 360 can do it — but it’s way more work than it should be.

> ✅ This tool currently supports only STL files (others untested).  
> 🧪 I’ve used it on scans up to **750 MB**; it gets a bit slow, but works.

---

## 🔧 Built With

- **Python 3.10**
- Required packages:

  ```bash
  pip install pyvista vtk numpy
  ```

---

## 💻 How It Works

### Step 1: Choose Your Version

- `commandlineversion.py`: basic version — looks for a file called `INPUTSTLFILE.stl` in the same folder.
- `originaligner.py`: drag-and-drop version. Can be compiled into a standalone `.exe` with:

  ```bash
  pyinstaller --onefile originaligner.py
  ```

### Step 2: Using the Tool

When you open an STL file:

1. A 3D viewer opens (rotate with your mouse).
2. **Select at least 3 points** to define the XY plane:
   - Hover your mouse over the model.
   - Press `P` or right-click to select a point.
   - A blue dot will appear.
3. **Keyboard controls:**
   - `B`: Undo the last selected point.
   - `1–9`: Remove a specific point by number.
   - `Space`: Confirm selection and align to the XY plane.

> Once confirmed, the file is saved as `alignedXY.stl`.

4. Close the viewer to continue to **Step 2: X-axis alignment**.
5. Select **2 or more points** that define a line you want to be parallel to the X-axis (top-down rotation).
   - Same controls apply (`P`, `B`, `1–9`, `Space`).
   - Uses a best-fit average direction vector.

> Once confirmed, the final result is saved as `alignedXYZ.stl`.

---

## ⚠️ Known Issues

- Sometimes the selected point (blue sphere) doesn’t appear — it may be on the backside of the model.
  - Try undoing and rotating your view before selecting again, having a steep angle in the view to what you want to click helps a lot i find.
- Large STL files may make the app slow, but it should still work.

---

## 🤖 About

This tool was quickly built with help from **ChatGPT**.  
Feel free to fork it, improve it, or build on it!
