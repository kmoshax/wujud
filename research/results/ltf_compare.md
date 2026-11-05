# L-LTF or HT-LTF for the CSI (2026-11-05)

The boards can send the HT-LTF part of the CSI instead of the L-LTF (`board.sh set csi_ltf=1`; same
64-bin line, the app reads it unchanged). Three 40 s recordings of the same four boards in the taped
bedroom, one after the other, measured with `research/tools/ltf_noise.py` (median packet-to-packet
change of the amplitude, each packet divided by its mean; lower is cleaner):

| recording | CSI part | links | median change |
|---|---|---|---|
| `rec/ltf_a_lltf.csv` | L-LTF | 12 | 0.0259 |
| `rec/ltf_b_htltf.csv` | HT-LTF | 12 | 0.0284 |
| `rec/ltf_c_lltf_again.csv` | L-LTF again | 11 | 0.0300 |

The two L-LTF recordings differ more than L-LTF and HT-LTF do: the room (a window opened, a person
at the desk) moves the number more than the choice of field. HT-LTF is not cleaner on these boards,
so the default stays L-LTF, which the models were trained on.
