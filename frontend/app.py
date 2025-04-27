from flask import (Flask, render_template, request, redirect,
                   url_for, session, send_file, send_from_directory, abort, jsonify)
import os
import pathlib
import requests
import json
import os
import io
from PIL import Image, ImageOps, ImageChops 
import base64
from flask_session import Session
import uuid, tempfile, shutil                                   # new utils
from recommendation import run_async_ai
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-only-change-me2")

app.config.update(
    SESSION_TYPE="filesystem",
    SESSION_FILE_DIR=os.path.join(tempfile.gettempdir(), "flask_sessions"),
    SESSION_PERMANENT=False,
)
Session(app)

# ─── helpers ────────────────────────────────────────────────────────────────

MASK_DIR = os.path.join('static', 'mask_cache')
os.makedirs(MASK_DIR, exist_ok=True)

def _masks_dict():
    """Return the per-image mask store held in session (create if absent)."""
    return session.setdefault("masks", {})   

def _add_mask(image_rel, data_uri):
    # write the mask to disk once
    uid   = uuid.uuid4().hex
    fname = f"{uid}.png"
    path  = os.path.join(MASK_DIR, fname)

    _, b64 = data_uri.split(",", 1)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))

    # remember *just* the relative path
    mdict = session.setdefault("masks", {})
    mdict.setdefault(image_rel, []).append(fname)
    session.modified = True

def _build_overlay(image_rel):
    fnames = session.get("masks", {}).get(image_rel)
    if not fnames:
        return None

    src_img = Image.open(os.path.join("static", image_rel)).convert("RGBA")
    agg     = Image.new("L", src_img.size, 0)

    for fname in fnames:
        m = Image.open(os.path.join(MASK_DIR, fname)).convert("L")
        agg = ImageChops.lighter(agg, m)          # pixel-wise max

    overlay = Image.new("RGBA", src_img.size, (0, 0, 255, 120))
    return Image.composite(overlay, src_img, agg)

def _pop_mask(image_rel):
    mdict = session.get("masks", {})
    if mdict.get(image_rel):
        fname = mdict[image_rel].pop()            # remove newest file name
        try:                                      # delete the mask file itself
            os.remove(os.path.join(MASK_DIR, fname))
        except FileNotFoundError:
            pass
        if not mdict[image_rel]:
            del mdict[image_rel]
        session.modified = True

def _overlay_bytes(image_rel):
    """PNG bytes of aggregated overlay *or* None if no masks."""
    comp = _build_overlay(image_rel)
    if comp is None:
        return None
    buf = io.BytesIO()
    comp.save(buf, format="PNG")
    buf.seek(0)
    return buf
# ────────────────────────────────────────────────────────────────────────────


UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # clear out any previous masks/labels
        session.pop('masks', None)
        session.pop('labels', None)
        session.modified = True
        label_text = request.form.get('labels', '').strip()

        # 2️⃣  save the uploaded images (unchanged)
        for f in request.files.getlist('folder_input'):
            if f.filename.lower().endswith(('.jpg', '.jpeg')):
                dst = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                f.save(dst)

        # 3️⃣  launch the OpenAI logic and PRINT the result
        if label_text:
            final_names = run_async_ai(label_text)
            print("\n======= FINAL NAMES FROM AI =======")
            for fn in final_names:
                print(fn)
            print("===================================\n")

        return redirect(url_for('slideshow_page'))

    return render_template('upload.html')


@app.route('/slideshow')
def slideshow_page():
    base   = pathlib.Path(app.config['UPLOAD_FOLDER'])
    images = [str(p.relative_to('static')) for p in base.rglob('*')
              if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    # rating logic unchanged (dummy 100) … omitted for brevity …
    labels  = session.get('labels', [])
    ratings = [100]*len(images)
    images  = [img for _, img in sorted(zip(ratings, images),
                                        key=lambda t: t[0], reverse=True)]

    # aggregated overlay (if masks already exist for first image)
    first_overlay = _overlay_bytes(images[0])
    overlay_duri  = (None if first_overlay is None else
                     "data:image/png;base64," + base64.b64encode(first_overlay.getvalue()).decode())

    return render_template('slideshow.html',
                           images=images, index=0, overlay_data=overlay_duri)

@app.route('/segment', methods=['POST'])
def segment():
    # --- 1. parse form data --------------------------------------------------
    base      = pathlib.Path(app.config['UPLOAD_FOLDER'])
    images    = [str(p.relative_to('static')) for p in base.rglob('*')
                 if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    index     = int(request.form['index'])
    image_rel = request.form['image']
    x, y      = int(request.form['x']), int(request.form['y'])
    img_path  = os.path.join('static', image_rel)

    # --- 2. call external segmentation service ------------------------------
    resp = requests.post(
        "https://point_api.kesava.lol/segment", verify=False,
        files={'image': open(img_path, 'rb')},
        data={'points': json.dumps([[x, y]]),
              'point_labels': json.dumps([1]),
              'multimask_output': 'false'},
        timeout=180
    )
    resp.raise_for_status()
    mask_datauri = resp.json()['masks_png'][0]       # first mask

    # --- 3. store mask -------------------------------------------------------
    _add_mask(image_rel, mask_datauri)

    # --- 4. build aggregated overlay & inline it ----------------------------
    overlay_buf  = _overlay_bytes(image_rel)
    overlay_duri = "data:image/png;base64," + base64.b64encode(overlay_buf.getvalue()).decode()

    return render_template('slideshow.html',
                           images=images, index=index, overlay_data=overlay_duri)

@app.route('/generate_route', methods=['POST'])
def generate_route():
    payload    = request.get_json() or {}
    datasource = payload.get('datasource_name')
    if datasource not in ('regensburg', 'munich'):
        return jsonify({'error': 'datasource_name must be "regensburg" or "munich"'}), 400

    masks_dict = session.get('masks', {})

    # ─── STALE-CACHE CLEANUP ────────────────────────────────────────────
    changed = False
    for image_rel in list(masks_dict.keys()):
        valid = []
        for fname in masks_dict[image_rel]:
            if os.path.exists(os.path.join(MASK_DIR, fname)):
                valid.append(fname)
            else:
                changed = True
        if valid:
            masks_dict[image_rel] = valid
        else:
            del masks_dict[image_rel]
            changed = True
    if changed:
        session['masks']     = masks_dict
        session.modified     = True
    # ────────────────────────────────────────────────────────────────────

    saved = {}

    MAX_BATCH = 256

    for image_rel, fnames in masks_dict.items():
        # 1) build the full lists
        u_list, v_list, names = [], [], []
        for fname in fnames:
            arr = np.array(Image.open(os.path.join(MASK_DIR, fname)).convert('L'))
            ys, xs = np.nonzero(arr)
            u_list.extend(xs.tolist())
            v_list.extend(ys.tolist())
            names.extend([os.path.basename(image_rel)] * len(xs))

        if not u_list:
            continue  # nothing to do
        all_points = []
        total = len(u_list)
        for start in range(0, total, MAX_BATCH):

            end = start + MAX_BATCH
            
            batch_body = {
                'imageName':        [image_rel.split("/")[-1]],
                'u':                u_list   [start:end],
                'v':                v_list   [start:end],
                'datasource_name':  datasource
            }
            print(batch_body)

            resp = requests.post(
                "https://pixeltranslate.kesava.lol/pixel-to-latlon/batch",
                json=batch_body,
                timeout=180
            )
            resp.raise_for_status()
            pts = resp.json()
            all_points.extend(pts)

        # 3) save the concatenated result
        base      = os.path.splitext(os.path.basename(image_rel))[0]
        out_fname = f"{base}_segmented_points.npy"
        out_path  = os.path.join(app.config['UPLOAD_FOLDER'], out_fname)
        np.save(out_path, np.array(all_points, dtype=object), allow_pickle=True)

        saved[image_rel] = out_fname

    return jsonify({'ok': True, 'saved_files': saved})



# ─── new: overlay image endpoint ────────────────────────────────────────────
@app.route('/overlay')
def overlay_route():
    image_rel = request.args.get('image')
    if not image_rel:
        abort(400)

    buf = _overlay_bytes(image_rel)
    if buf is None:                                    # no masks → original file
        return send_from_directory('static', image_rel)

    return send_file(buf, mimetype='image/png',
                     as_attachment=False, download_name='overlay.png')

# ─── new: undo mask endpoint ────────────────────────────────────────────────
@app.route('/undo', methods=['POST'])
def undo():
    image_rel = request.json.get('image')
    if not image_rel:
        return jsonify({"error": "image required"}), 400
    _pop_mask(image_rel)
    return jsonify({"ok": True}), 200

@app.route('/route')
def route_page():
    return render_template('route.html')

@app.route('/maskcount')
def maskcount():
    img = request.args['image']
    n   = len(session.get('masks', {}).get(img, []))
    return jsonify(count=n)


if __name__ == '__main__':
    app.run(debug=True, port=5000)