"""Build the explainer page from the results files: python explainer/build.py -> explainer/hallsense_explainer.html"""
import json, os
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H)
J = lambda p: json.load(open(os.path.join(ROOT, p)))

s2r = J("results/room_sim2real.json")["all"]
real = dict(motion_recall=s2r["sim_only"]["motion_recall"], motion_fp=s2r["sim_only"]["motion_fp"],
            auc_sim=s2r["sim_only"]["auc_empty_still"], auc_rec=s2r["sim_only"].get("auc_empty_still_recording", float("nan")),
            acc_sim=s2r["sim_only"]["acc"], auc_real=s2r["real_trained"]["auc_empty_still"], acc_real=s2r["real_trained"]["acc"],
            real_motion_recall=s2r["real_trained"]["motion_recall"], real_motion_fp=s2r["real_trained"]["motion_fp"])

r3 = J("results/results3d.json")
pts = [[r["true"], round(r["count"], 1), round(r["rti_count"], 1)] for r in r3["rows"]]
sweep = [[int(n), v["mae_pct_of_seats"]] for n, v in sorted(r3.get("sweep", {}).items(), key=lambda kv: int(kv[0]))]
hall = dict(mae=r3["test"]["ours"]["mae"], mae_pct=r3["test"]["ours"]["mae_pct_of_seats"], rti_mae_pct=r3["test"]["rti"]["mae_pct_of_seats"],
            auc_sparse=r3["test"]["ours"]["auc"]["sparse(<25%)"], points=pts, sweep=sweep)

re = J("results/room_eval.json")
lab = {"esp1_random": "بوردة واحدة، مكان عشوائي", "esp1_smart": "بوردة واحدة، الخط معدّي على السرير/المكتب",
       "esp3_random": "٣ بوردات، أماكن عشوائية", "esp3_smart": "٣ بوردات، واحدة منهم معدّية على السرير"}
rows = []
for k in ("esp1_random", "esp1_smart", "esp3_random", "esp3_smart"):
    if k in re:
        v = re[k]
        loc = "مش متاح ببوردة واحدة" if k.startswith("esp1") else f"{v.get('loc_median_m', float('nan')):.1f} م (ضعيف، قريب من التخمين)"
        rows.append([lab[k], v["acc3"], v["acc5"], v["breath_median_err_confident"], loc])
room = dict(rows=rows, breath_err3=re.get("esp3_random", re.get("esp1_smart"))["breath_median_err_confident"])

w = J("results/room_world.json")
curve = w["through_wall"]
limit = max([c["wall_db"] for c in curve if c["still_auc"] >= 0.8] or [0])
wm = w["what_moved"]

failures = [
    ["تتبّع لحظات الحركة: نفك في كل ثانية مين اللي اتحرك (sparse decoding)", "compressed sensing / الطب", "ما حسّنش تحديد الكراسي", "no"],
    ["البوردات على أطراف الصفوف بدل الحيطان", "هندسة التصوير المقطعي", "خطأ العدّ زاد من ١٠٪ لـ ١٤٪", "no"],
    ["معايرة القاعة بعدّة يدوية واحدة", "calibration", "الخطأ زاد (من ٢٤٪ لـ ٣٤٪): الخطأ مش انحياز ثابت", "no"],
    ["كاشف تنفس بيجمع كل الـ subcarriers (Patwari GLRT) على الداتا الحقيقية", "الطب الحيوي / رادار العلامات الحيوية", "AUC 0.70، مش أحسن من الموجود", "mid"],
    ["البوردات ترسم خريطة لنفسها من قوة الإشارة (MDS-MAP)", "SLAM / شبكات الحساسات", "خطأ ~٤ م في قاعة ١٣ م، مش صالح (محتاج FTM)", "no"],
    ["تحديد مكانك في الأوضة من غير معايرة", "Fresnel matching + تعلّم", "١٫٢٥ م مقابل ١٫١٤ م للتخمين، فشل", "no"],
    [f"التفريق بين الإنسان والقطة والمكنسة والمروحة والباب", "Wi-MoID (IEEE IoT-J 2024)",
     f"جزئي: الإنسان اتكشف {wm['human_recall'] * 100:.0f}٪، والأجسام اتحسبت إنسان {wm['human_false_alarm_from_objects'] * 100:.0f}٪ (البحث بيقول ٩٧٪)", "mid"],
]
pe, sc = J("results/phase_exp.json"), J("results/sim_calibrate.json")
failures += [
    ["الـ phase بعد تنضيفه من أخطاء الساعة عشان يبيّن التنفس", "PhaseBeat / phase sanitization",
     f"ما حسّنش على الـ selection ({pe['amplitude + phase']['selection']['still_vs_empty_auc']:.3f} مقابل {pe['amplitude (current)']['selection']['still_vs_empty_auc']:.3f})", "no"],
    ["نظبط المحاكي لحد ما التنفس يبان بنفس نسبة الواقع", "domain calibration",
     f"فرق الرؤية قلّ ({min(g['sim_visibility'] for g in sc['grid']) * 100:.0f}٪ مقابل {sc['target_visibility_train'] * 100:.1f}٪)، لكن الـ AUC ما اتحسنش "
     f"({sc['calibrated sim, amplitude']['selection']['still_vs_empty_auc']:.3f} مقابل {sc['baseline']['selection']['still_vs_empty_auc']:.3f})", "no"],
    ["اللابتوب يسمع beacons الراوترين (من غير ما يتوصل) ويكشف الثبات ويحدد المكان", "passive RSSI sensing",
     "المشي اتكشف ١٠٩ من ١٢٠ ثانية؛ الثبات مقابل الفاضي ما اتفرقوش؛ المكان تقريبي (على أنهي خط)", "mid"],
]
sc_ = J("results/self_calib.json")["summary"]; fa = J("results/occupancy_real_fa.json"); osim = J("results/occupancy_sim.json")["test"]
rep = J("results/occupancy_engine_replay.json")
still_ok = [sum(1 for r in v if r[1] == "still" and r[2] == "still") for k, v in rep.items() if "realistic" in k]
occupancy = [
    ["معايرة ثابتة: نفس الجلسة (رقم مضروب، الجلسات بتفرق)", "ESPectre", f"AUC {sc_['same_session']['all features']:.2f}"],
    ["معايرة ثابتة: يوم فاضي ← يوم فاضي تاني (الاختبار الأمين)", "ESPectre", f"AUC {sc_['cross_session']['all features']:.2f}"],
    ["occupancy latch: إنذار كاذب على أوض فاضية حقيقية", f"ESPectre، {sum(fa['per_recording_minutes']):.0f} دقيقة", f"{fa['false_present_rate']['0.4'] * 100:.0f}٪"],
    ["occupancy latch: شخص ثابت بعد ما دخل (فاضية ← مشي ← ثابت)", "جلستين حقيقيتين", " و ".join(f"{n}/180 ث" for n in still_ok) + " (والـ classifier لوحده نفس الرقم)"],
    ["occupancy latch مقابل الـ classifier في أيام محاكاة", "100 أوضة جديدة، محاكي متشائم",
     f"{osim['occupancy_latch']['accuracy']:.2f} مقابل {osim['per_window_classifier']['accuracy']:.2f}"],
]
rv = J("results/ruview_compare_native.json")
ruview = [["إنذار كاذب بحركة", f"{rv['HallSense (sim-only)']['motion_false_alarm'] * 100:.1f}٪", f"{rv['RuView edge']['motion_false_alarm'] * 100:.1f}٪"],
          ["ترتيب الحركة (AUC)", f"{rv['HallSense (sim-only)']['motion_auc']:.2f}", f"{rv['RuView edge']['motion_auc']:.2f}"],
          ["ثابت مقابل فاضي (AUC)", f"{rv['HallSense (sim-only)']['still_vs_empty_auc']:.2f}", f"{rv['RuView edge']['still_vs_empty_auc']:.2f}"],
          ["عدد التسجيلات / النوافذ", f"{rv['n_recordings']} / {rv['n_windows']}", f"{rv['n_recordings']} / {rv['n_windows']}"]]
data = dict(samples=json.load(open(os.path.join(H, "sim_samples.json"))),
            results=dict(real=real, hall=hall, room=room, wall=dict(curve=curve, limit_db=limit)), failures=failures, ruview=ruview, occupancy=occupancy)
import base64, io
from PIL import Image
def img(path):
    im = Image.open(os.path.join(ROOT, path)).convert("RGB"); im.thumbnail((1400, 1400))
    b = io.BytesIO(); im.save(b, "JPEG", quality=82); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
html = open(os.path.join(H, "template.html"), encoding="utf-8").read().replace("__DATA__", json.dumps(data, ensure_ascii=False))
html = html.replace("__IMG_HALL__", img("results/dashboard_hall3d.png")).replace("__IMG_ROOM__", img("results/dashboard_room3d.png"))
open(os.path.join(H, "hallsense_explainer.html"), "w", encoding="utf-8").write(html)
print("built", len(html) // 1024, "KB; wall limit", limit, "dB")
