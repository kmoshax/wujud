# الفيزيا ورا المحاكي: بيحسب إيه بالظبط؟

الصفحة دي للي عايز يتأكد إن المحاكاة مش "رسمة حلوة". كل معادلة هنا موجودة في الكود زي ما هي، وتحت كل واحدة اسم الملف والدالة اللي بتنفّذها. والأرقام هي القيم الحقيقية اللي المحاكي بيسحب منها عشوائيًا. الترتيب ماشي زي الإشارة نفسها: من الهوائي، للأوضة، للجسم، لحد الـ ESP32، وبعدين الموديل.

> أسامي الملفات هنا من النسخة الأصلية بالـ Python (`research/wujud/`). نفس المعادلات منقولة لـ Rust في `crates/sim`، والـ README بتاعه بيقول كل معادلة اتنقلت فين.


## أ١. القناة: الـ CSI عبارة عن مجموع مسارات

البوردة بتقيس لكل subcarrier رقم مركّب واحد اسمه H(f,t). الرقم ده بيقول الموجة وصلت أضعف بقد إيه واتأخرت بقد إيه. ولأن الإشارة بتوصل من كذا طريق في نفس الوقت، الرقم اللي بنقيسه هو مجموع كل الطرق دي:

```
H(f, t) = Σ_k F_k(t) · g_k · λ / (4π L_k) · e^(−j2π f L_k/c) + H_clutter(f) + H_body scatter(f, t)
```

- **L<sub>k</sub>**: طول المسار رقم k بالمتر، والتأخير بتاعه τ<sub>k</sub> = L<sub>k</sub>/c

- **λ/(4πL)**: فقد الفضاء الحر (Friis) في صورة amplitude. عند 2.437 GHz الطول الموجي λ = c/f<sub>c</sub> ≈ 0.123 م

- **g<sub>k</sub>**: معامل الانعكاس المركّب: 1 للخط المستقيم، و−0.25…−0.6 للحيطان، و−0.5 للأرض، و−0.35 للسقف

- **F<sub>k</sub>(t)**: معامل الحجب من الأجسام على المسار ده (بنشرحه في أ٤). من غير ناس بيبقى 1

- **f**: تردد الـ subcarrier: f = f<sub>c</sub> + n·Δf، حيث Δf = 312.5 kHz. المدرج بيستخدم الـ 52 subcarrier بتوع L-LTF (n = ±1…±26)، والأوضة بتستخدم الـ 56 بتوع HT-LTF (n = ±1…±28)

**الفكرة ببساطة:** عرض النطاق حوالي 17.5 MHz، يعني دقة التأخير حوالي 1/17.5MHz ≈ 57 ns، وده يساوي ~17 م من المسافة. عشان كده الـ ESP32 مش بيقدر يفصل المسارات عن بعض جوه أوضة. اللي بيشوفه هو "نمط تداخل" على الـ subcarriers، والإنسان بيغيّر النمط ده. وده بالظبط اللي بنحاكيه.

الكود: `wujud/config.py · wujud/physics3d.py → path_response(), link_channel()`

## أ٢. الانعكاسات: طريقة الصور (Image method)

عشان نلاقي نقطة الانعكاس على حيطة، بنعكس المرسِل في الحيطة كأنها مراية. وبعدين نوصّل خط مستقيم من "صورة" المرسِل للمستقبِل. النقطة اللي الخط بيقطع فيها الحيطة هي نقطة الانعكاس. لو الحيطة عند x = x<sub>w</sub>:

```
a′ = (2x_w − a_x, a_y, a_z), P = a′ + t (b − a′), t = (x_w − a′_x) / (b_x − a′_x), L = |a − P| + |P − b| = |b − a′|
```

بنعمل ده لأربع حيطان والأرض والسقف، فيطلع لكل لينك **7 مسارات هندسية**: الخط المستقيم و6 انعكاسات من أول رتبة. الطريقة دي أصلها من صوتيات الأوض [1]. وكل مسار بيتخزن كخط مكسور ثلاثي الأبعاد، عشان نقدر بعدين نشوف لو حد واقف على أي جزء منه.

الكود: `wujud/physics3d.py → link_paths3()`

## أ٣. الـ multipath المنتشر: مسارات عشوائية لكن ليها هندسة

الانعكاسات من أول رتبة مش كفاية. في الواقع الطاقة بتلف الأوضة كلها. نماذج القنوات العشوائية المعتادة بتضيف "ضوضاء ملوّنة" ملهاش مكان، وده معناه إن ما فيش حد يقدر يأثر عليها. إحنا بدل كده بنختار N نقطة عشوائية v<sub>k</sub> جوه الأوضة، ونعمل مسار a → v<sub>k</sub> → b بقدرة متوزعة حسب profile تأخير أُسّي:

```
P_k ∝ exp(−(L_k − d) / (c·τ_rms)), Σ_k P_k = (λ/4πd)^(2) · 10^(−K/10), phase_k ~ U(0, 2π)
```

- **N**: 24 مسار في الأوضة، و8 لكل لينك في المدرج (عشان سرعة الحساب)

- **τ<sub>rms</sub>**: انتشار التأخير، من 15 لـ 40 ns

- **K**: معامل Rician (قد إيه الخط المستقيم أقوى من المنتشر)، من −3 لـ +6 dB

**ليه ده مهم؟** لأن كل مسار منتشر ليه مكان، فحد ماشي في أي حتة في الأوضة هيحجب كام مسار منهم. الحتة دي اتضافت بعد ما الداتا الحقيقية وضّحت إن الحركة بتبان من أي مكان، والمحاكي القديم ما كانش بيشوفها.

الكود: `wujud/physics3d.py → diffuse_paths()`

## أ٤. الجسم بيحجب: حيود Fresnel-Kirchhoff على عائق مستطيل

نصف قطر منطقة Fresnel رقم n عند نقطة بتبعد d<sub>1</sub> عن المرسِل وd<sub>2</sub> عن المستقبِل:

```
r_n = √( n λ d_1 d_2 / (d_1 + d_2) )
```

للينك طوله 4 م، نصف القطر في النص r<sub>1</sub> ≈ 0.35 م، وده قريب جدًا من عرض جسم إنسان. وللينك طوله 12 م في مدرج، r<sub>1</sub> ≈ 0.61 م. يعني الجسم مش "نقطة صغيرة" بالنسبة للموجة، وعشان كده لازم نحسب الحيود.

الجسم بيتمثّل كصندوق: نص عرض a (الكتاف)، ونص عمق b (الصدر، والتنفس بيغيّره)، ومن ارتفاع z<sub>0</sub> لـ z<sub>1</sub>، ومتلفّ بزاوية yaw. وكل جزء من كل مسار بيتحسب في إحداثيات عمودية عليه (h أفقي وv رأسي). المجال اللي بيعدّي من عائق مستطيل بيمتص بنسبة η:

```
E/E_0 = 1 − η · (j/2) · I(u_1, u_2) · I(v_1, v_2), I(α, β) = ∫_α^(β) e^(−jπt²/2) dt = [C(β)−C(α)] − j[S(β)−S(α)]
```

```
u_1,2 = (h ∓ w_h)·k_F, v_1,2 = (v ∓ w_v)·k_F, k_F = √( 2L / (λ d_1(L − d_1)) )
```

- **C, S**: تكاملات Fresnel (scipy.special.fresnel)

- **h, v**: إزاحة مركز الجسم عن المسار، أفقيًا ورأسيًا

- **w<sub>h</sub>, w<sub>v</sub>**: نص عرض وارتفاع إسقاط الصندوق على المستوى العمودي على المسار (بتتحسب من a وb وyaw)

- **η**: نسبة الامتصاص: 0.75–0.97 للطالب القاعد، و0.8–0.95 لشخص في الأوضة، و0.6 للقطة

لما يكون فيه أكتر من جسم، المعاملات بتتضرب في بعض. وأي جسم يبعد أكتر من 3.5 وحدة Fresnel عن المسار بيتشال من الحساب عشان السرعة. التكامل ده منفصل في الاتجاهين، فبيدّي النتيجة الدقيقة لحافتين مستقيمتين (knife-edge) [2][3]، وأسرع بكتير من ray-tracing كامل. وده نفس الحساب اللي ورا الرسمة التفاعلية في القسم ٢-أ.

الكود: `wujud/physics3d.py → segment_shadow(), _I()`

## أ٥. التنفس والحركات الصغيرة: من المليمتر للزاوية

الصدر متمثّل كنقطة بتعكس (معادلة الرادار الثنائي)، وكمان الإيد ليها نقطة تانية:

```
H_scat(f,t) = λ√σ / ((4π)^(3/2) d_1 d_2) · e^(−j2π f L_0/c) · e^(−j2π f_c ΔL(t)/c)
```

σ هو مقطع الرادار: 0.3–0.8 م² لصدر شخص في الأوضة، و0.1–0.5 للطالب القاعد. لما الصدر يتحرك مسافة x، طول المسار بيتغير بحوالي ΔL ≈ 2x·cos(β/2)، حيث β هي الزاوية بين المرسِل والمستقبِل من عند الجسم. والطور بيتغير بـ:

```
Δφ = 2π ΔL / λ ⇒ x = 5 mm, ΔL ≈ 10 mm → Δφ ≈ 0.51 rad ≈ 29°
```

التنفس متمثّل بموجة جيبية x(t) = A·sin(2π f<sub>b</sub> t + φ<sub>0</sub>)، وبيحرّك نقطة الصدر وبيغيّر عمق الصندوق b في نفس الوقت. يعني بيأثر على الانعكاس والحجب مع بعض.

- **A**: 4–8 مم في الأوضة، و3–6 مم للطالب القاعد

- **f<sub>b</sub>**: 0.2–0.33 Hz، يعني 12 لـ 20 نفس في الدقيقة

- **sway**: تمايل الجسم: عملية Ornstein-Uhlenbeck بـ σ = 4 مم وτ = 3 ث

- **fidgets**: حركات صغيرة على شكل Poisson. معدّل كل طالب lognormal بوسيط "حركة كل 20 ث"، ومدة كل حركة 0.8–4 ث، وإزاحتها 2–15 سم

- **walk**: المشي بسرعة 0.5–1.2 م/ث لنقط عشوائية. الدكتور بيمشي قدام المدرج بـ 0.3–0.9 م/ث وبيقف فترات أُسّية

الكود: `wujud/physics3d.py → scatter_series() · wujud/scenes3d.py → room_person(), seated_bodies(), lecturer(), _ou()`

### ليه التنفس ساعات بيبان وساعات لأ؟ (نموذج الـ phasor)

في كل subcarrier، الإشارة عبارة عن متّجه ثابت H<sub>s</sub> (كل المسارات اللي ما بتتحركش) زائد متّجه صغير بيلفّ مع النفس. والـ ESP32 بيقيس الطول بس، |H|. لو اللفّة حصلت والمتّجه الصغير في نفس اتجاه H<sub>s</sub> أو عكسه (θ₀ ≈ 0° أو 180°)، الطول بيبقى عند قمة أو قاع، والتغيير فيه من الدرجة التانية بس، فالتنفس "بيختفي". أما لو كان عمودي عليه (θ₀ ≈ 90°)، اللفّة بتطوّل وتقصّر المجموع مباشرة، والتنفس بيبان أوضح ما يمكن. ودي بالظبط نظرية Fresnel zone للتنفس [7][8]. جرّب بنفسك:

## أ٦. امتصاص الناس: معادلة Sabine للراديو

في الصوتيات، زمن الصدى بيتحسب من T<sub>60</sub> = 0.161·V/A، حيث A = Σ S<sub>i</sub> α<sub>i</sub> هي مساحة الامتصاص الكلية [4]. واتثبت إن نفس العلاقة شغالة للمجال الكهرومغناطيسي المنتشر جوه الأوض [5]. وفي الحالة المستقرة، القدرة المنتشرة بتتناسب عكسيًا مع A. فلما يدخل N شخص، كل واحد بيمتص ~0.4 م²:

```
A_0 = α·S, S = 2WD + 2(W+D)H, α ~ U(0.1, 0.3) ⇒ P_N/P_0 = A_0 / (A_0 + 0.4N), |H| scaled by √(A_0/(A_0+0.4N))
```

المعامل ده بيتضرب في الجزء "المبعثر": انعكاسات العفش والكراسي والأجسام. أما المسارات الهندسية فبتتأثر بالحجب المباشر (أ٤). مثال: مدرج 13×12×4 م وα = 0.2، يعني A<sub>0</sub> ≈ 102 م². لو دخل 120 طالب، القدرة المنتشرة بتقل حوالي 1.7 dB. ولو قلبنا المعادلة بنقدر نقدّر العدد مباشرة:

```
N̂ ≈ (α S / 0.4) · (10^(−Δ/10) − 1), Δ = shadow (dB, session − calibration)
```

الـ feature دي (S·(10<sup>−Δ/10</sup>−1)) داخلة فعلًا في موديل العدّ.

الكود: `wujud/sim3d.py → _absorb(), simulate_hall3() · wujud/model.py → scene_vector()`

## أ٧. ورا الحيطة: توهين + "فلاش" ثابت

```
H_wall(f,t) = 10^(−L_w/20) · H_room(f,t) + H_flash(f), H_flash = Σ_m=1..6 g_m e^(−j2π f (d/c + τ_m))
```

L<sub>w</sub> هو توهين الحيطة من 0 لـ 40 dB، وقيمه مأخوذة من Wi-Vi [6]. والـ "flash" هو انعكاسات من على وش الحيطة وتسريب حواليها: 6 مسارات ثابتة بتأخير زيادة τ<sub>m</sub> ∈ [5, 60] ns، وكسب g<sub>m</sub> مركّب Gaussian على مقياس ~0.35× الفضاء الحر. الفلاش ما بيتأثرش بالحيطة، فكل ما الحيطة تتخن، تعديل الشخص بيغرق تحت الفلاش والضوضاء. ومن هنا بييجي الحد العملي اللي ~18 dB للتنفس.

الكود: `wujud/sim3d.py → simulate_room(wall_db=…)`

## أ٨. عيوب الـ ESP32: من رقم مثالي لـ int8

القناة النظيفة بتعدّي على "جهاز استقبال" محاكى، بنفس ترتيب الحاجات في الهاردوير الحقيقي:

```
Y = H·10^(P_tx/20) + n, n ~ CN(0, σ²), σ² = N_0 + P_sig·10^(EVM/10)
```

```
RSSI = round(10 log_10 P_rx + b(t) + 𝒩(0, 0.5)), IQ = clip(round(Y · g_AGC/√P_rx), −127, 127)
```

- **N<sub>0</sub>**: أرضية الضوضاء الحرارية: −94 dBm

- **P<sub>tx</sub>**: قدرة الإرسال: 4–16 dBm، مختلفة لكل بوردة. وفي المدرج فيه drift بين المعايرة والمحاضرة σ = 0.25 dB

- **EVM**: خطأ تقدير القناة: −29…−23 dB، بيكبر مع الإشارة القوية

- **g<sub>AGC</sub>**: الـ AGC بيستهدف amplitude من 12–24 وحدة، وبيتهز من packet للتانية بـ ±1 dB (lognormal)

- **b(t)**: انحياز الـ RSSI: بيتنط حوالي 3 مرات في الدقيقة (Poisson) بـ σ = 0.6 dB

- **loss / interf**: packets بتضيع بنسبة 1–8%، و0.5–3% من الـ packets بيبقى فيها تداخل من −12 لـ 0 dB نسبةً للإشارة، وتوقيتها بيتهز ±2 ms

**ملحوظة صريحة:** المحاكي **ما بيحاكيش** الـ CFO/STO (إزاحة التردد والتوقيت اللي بتخلّي الطور عشوائي من packet للتانية). ده مقصود، لأن كل الـ pipeline بيستخدم الـ amplitude بس، |CSI|، وبيطبّعه لكل packet على حدة. يعني الطور العشوائي ده ما بيوصلش لأي feature أصلًا، فمش محتاجين نحاكيه.

الكود: `wujud/sim3d.py → measure(), _radio()`

## أ٩. الـ features والموديلات

**الأوضة (لينك واحد، CSI):** بنشيل الـ AGC بتطبيع كل packet على متوسطها: ã<sub>k</sub>(t) = |CSI<sub>k</sub>| / mean<sub>k</sub>|CSI<sub>k</sub>|. بعدين بنعيد التقطيع على شبكة 20 Hz، وناخد شبابيك 20 ث بخطوة 5 ث، ونحوّل لـ dB ونشيل المتوسط، ونطلّع طيف Hann لكل subcarrier:

```
R_band,k = log_10( P̄_k(band) / P̄_k(4–9.5 Hz) ), breath = 0.12–0.6 Hz, micro = 0.6–3 Hz
```

الإنسان مش بيتحرك بين 4 و9.5 Hz، فالنطاق ده بيقيس ضوضاء الجهاز نفسه، وكل حاجة بتبقى نسبة ليه. يعني الـ features مش بتعتمد على نوع البوردة ولا على بُعد الراوتر. بعد كده بناخد أحسن ⅛ من الـ subcarriers ونطلّع:

- الـ PCA على النسخة المتفلترة في نطاق التنفس، وتركيز طيف المكوّن الأول: ρ = max S(f) / Σ S(f). ومكان القمة × 60 هو معدل التنفس.

- features بتقول "إيه اللي اتحرك": خط طيفي (مروحة)، وخطوة (باب)، وانفجارات (إنسان)، ونسبة الـ subcarriers اللي اتأثرت.

- مؤشر GLRT للتنفس بيجمع كل الـ subcarriers، على طريقة Patwari [9].

المصنِّف هو Gradient-Boosted Trees (HistGradientBoosting، نفس فكرة LightGBM [20]) بأوزان متوازنة للفئات. وفي وضع اللابتوب نفس النطاقات بتتحسب على الـ RSSI بدل الـ CSI.

الكود: `wujud/roomfeat.py → amplitude_from_iq(), window_features(), extra_features() · wujud/rssifeat.py`

**المدرج (شبكة لينكات):** كل packet بيتحوّل لـ 13 مجموعة (4 subcarriers في كل واحدة) بالشكل a<sub>k</sub> = 10log<sub>10</sub>(p<sub>k</sub>/p̄) + RSSI. والـ packets اللي الـ RSSI بتاعها بيبعد أكتر من 8 dB عن الـ EMA بتترمي. ولكل لينك بنقارن المحاضرة بالمعايرة اللي اتعملت والقاعة فاضية:

```
shadow_ℓ = median_session(a) − median_calib(a), motion_ℓ = log( σ_session / σ_calib )
```

وبعدين الحتة اللي بتخلّي الموديل يشتغل في أي قاعة: بدل ما نتعلم "لينك 3 + لينك 7 = كرسي 42"، اللي مش هيشتغل غير في نفس القاعة، بنوصف كل كرسي بعلاقته باللينكات من خلال **فرق طول المسار**:

```
ε_ℓ(p) = |a − p| + |p − b| − |a − b|, first Fresnel zone ⇔ ε < λ/2 ≈ 0.061 m, bins: 0, 1.5, 3.5, 7, 14, 28 cm
```

لكل كرسي، بنجمع features اللينكات اللي بتعدّي عليه في كل bin. الوصف ده معناه واحد في أي قاعة، وبأي عدد بوردات، وبأي توزيع ليها.

- **المرحلة 1:** GBDT بيطلّع P(الكرسي مشغول).

- **المرحلة 2:** نفس الحاجة، ومعاها احتمالات الجيران و"explaining away": مين غيرك ممكن يكون سبب التوهين على نفس اللينكات؟ المرحلة دي متدربة على احتمالات out-of-fold، عشان ما تتعلمش من تنبؤات شافتها قبل كده.

- **العدد:** N̂ = ½(Σ<sub>s</sub> p<sub>s</sub> + N̂<sub>reg</sub>). الجزء الأول مجموع احتمالات الكراسي، والتاني regressor على features القاعة كلها، ومنها مقلوب Sabine من أ٦.

والمقارنة كانت مع RTI الكلاسيكي [11]: x̂ = (WᵀW + αI)<sup>−1</sup>Wᵀy، حيث W<sub>ℓs</sub> = 𝟙[ε<sub>ℓs</sub> < λ<sub>e</sub>] / √d<sub>ℓ</sub>، وα وλ<sub>e</sub> والعتبة اتظبطوا على داتا التدريب.

الكود: `wujud/features.py · wujud/model.py → seat_descriptor(), context_features(), SeatModel, RTIBaseline · wujud/geometry.py → excess_path()`

## أ١٠. إزاي قسنا، ومعنى AUC

- **المدرج:** اتدرب على قاعات 2.5D و3D محاكاة، واتقاس على 20 قاعة 3D جديدة ما اتشافتش في التدريب (40 محاضرة). القاعات مختلفة في المقاس والميل وعدد الممرات وأماكن البوردات.

- **الأوضة:** أوض محاكاة جديدة، بعفش وأماكن بوردات مختلفة عن التدريب.

- **الداتا الحقيقية:** تسجيلات ESPectre [17]. استخدمنا التقسيمة بتاعتهم: الموديل اللي اتدرب على المحاكاة بس اتقاس على selection + holdout، ونفس المصنِّف اتدرب على الـ train الحقيقي كمرجع.

```
AUC = P( score(positive) > score(negative) ) = (1 / n_+n_−) Σ_i∈+ Σ_j∈− 𝟙[s_i > s_j]
```

يعني لو اخترت شباك فيه شخص ثابت وشباك تاني الأوضة فيه فاضية، AUC هو احتمال إن الموديل يدّي الأول درجة أعلى. 0.5 معناها تخمين عشوائي، و1 معناها مثالي [21]. ودي الطريقة اللي بتخلّي النتيجة مش معتمدة على عتبة معيّنة.

الكود: `evaluate3d.py · bench/room_eval.py · bench/room_sim2real.py · wujud/metrics.py`

## أ١١. ليه الـ sim2real اشتغل، وفين وقف

**اشتغل في الحركة** لأن الحركة بتحجب مسارات كتير وبتغيّر الـ θ₀ بتاع كل subcarrier أكتر من دورة كاملة، فبتبان في أي مكان وأي θ₀. ولأن الـ features كلها نسب لضوضاء الجهاز نفسه، الفرق بين المحاكي والهاردوير الحقيقي بيتلغي.

**ووقف عند الشخص الثابت** لسبب فيزيائي، مش عيب في الموديل. التنفس بيغيّر المسار ~1 سم بس، فعشان يبان لازم حاجتين مع بعض: (1) الصدر يكون جوه أول كام منطقة Fresnel لمسار قوي، و(2) الـ θ₀ تكون في حتة "حساسة" من الدايرة (شوف رسمة الـ phasor). في الداتا الحقيقية الشرطين دول بيتحققوا قليل: التنفس بان في ~11% بس من نوافذ الشخص الثابت. والمحاكي كان متفائل وقال ~50%، ولما ظبطناه يقرب من الواقع الموديل ما اتحسنش. ده بيأكد إن ده حد فيزيائي للينك الواحد. ومن هنا قاعدة التركيب: خلّي الخط يعدّي على السرير أو المكتب، أو استخدم كذا بوردة عشان على الأقل لينك واحد يقع في حتة حساسة. وده نفس الاستنتاج اللي وصل له [7] و[8] بالتجربة.

## المراجع

1. J. B. Allen and D. A. Berkley, "Image method for efficiently simulating small-room acoustics," J. Acoust. Soc. Am., vol. 65, no. 4, pp. 943–950, 1979. doi:10.1121/1.382599
2. M. Born and E. Wolf, Principles of Optics, 7th ed., Cambridge University Press, 1999, ch. 8 (Fresnel–Kirchhoff diffraction theory).
3. ITU-R Recommendation P.526, "Propagation by diffraction." itu.int/rec/R-REC-P.526
4. W. C. Sabine, Collected Papers on Acoustics, Harvard University Press, 1922.
5. J. B. Andersen, J. Ø. Nielsen, G. F. Pedersen, G. Bauch, and M. Herdin, "Room electromagnetics," IEEE Antennas and Propagation Magazine, vol. 49, no. 2, pp. 27–33, 2007. PDF
6. F. Adib and D. Katabi, "See through walls with WiFi!," Proc. ACM SIGCOMM, 2013. doi:10.1145/2486001.2486039 · PDF
7. H. Wang, D. Zhang, J. Ma, Y. Wang, Y. Wang, D. Wu, T. Gu, and B. Xie, "Human respiration detection with commodity WiFi devices: Do user location and body orientation matter?," Proc. ACM UbiComp, pp. 25–36, 2016. doi:10.1145/2971648.2971744
8. Y. Zeng, D. Wu, J. Xiong, E. Yi, R. Gao, and D. Zhang, "FarSense: Pushing the range limit of WiFi-based respiration sensing with CSI ratio of two antennas," Proc. ACM IMWUT, vol. 3, no. 3, 2019. doi:10.1145/3351279
9. N. Patwari, J. Wilson, S. Ananthanarayanan, S. K. Kasera, and D. Westenskow, "Monitoring breathing via signal strength in wireless networks," arXiv:1109.3898, 2011. arxiv.org/abs/1109.3898
10. N. Patwari et al., "Breathfinding: A wireless network that monitors and locates breathing in a home," arXiv:1302.3820, 2013. arxiv.org/abs/1302.3820
11. J. Wilson and N. Patwari, "Radio tomographic imaging with wireless networks," IEEE Transactions on Mobile Computing, 2010. PDF
12. S. Depatla, A. Muralidharan, and Y. Mostofi, "Occupancy estimation using only WiFi power measurements," IEEE J. Selected Areas in Communications, vol. 33, no. 7, 2015. doi:10.1109/JSAC.2015.2430272
13. G. Zhu, Y. Hu, B. Wang, C. Wu, X. Zeng, and K. J. R. Liu, "Wi-MoID: Human and nonhuman motion discrimination using WiFi with edge computing," IEEE Internet of Things Journal, vol. 11, no. 8, pp. 13900–13912, 2024.
14. Y. Shang, W. Ruml, Y. Zhang, and M. Fromherz, "Localization from mere connectivity," Proc. ACM MobiHoc, 2003. (MDS-MAP)
15. S. M. Hernandez and E. Bulut, "Lightweight and standalone IoT based WiFi sensing for active repositioning and mobility," Proc. IEEE WoWMoM, pp. 277–286, 2020. ESP32 CSI Tool: stevenmhernandez.github.io/ESP32-CSI-Tool
16. Espressif Systems, ESP-IDF Programming Guide, "Wi-Fi Channel State Information." docs.espressif.com
17. F. Pace, ESPectre: open-source ESP32 Wi-Fi motion/presence sensing with recorded CSI datasets. github.com/francescopace/espectre
18. ruvnet, RuView (formerly wifi-densepose): WiFi CSI sensing, pose and vital signs. github.com/ruvnet/RuView
19. "A survey on Wi-Fi sensing generalizability: Taxonomy, techniques, datasets, and future research prospects," arXiv:2503.08008, 2025. arxiv.org/abs/2503.08008
20. G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," NeurIPS, 2017; F. Pedregosa et al., "Scikit-learn: Machine learning in Python," JMLR, vol. 12, pp. 2825–2830, 2011.
21. J. A. Hanley and B. J. McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," Radiology, vol. 143, no. 1, pp. 29–36, 1982.
22. MIT Lincoln Laboratory, "Motion Under Rubble Measured Using Radar (MURMUR)." ll.mit.edu
23. Y. Mostofi lab (UCSB), counting a stationary crowd with WiFi. web.ece.ucsb.edu/~ymostofi
