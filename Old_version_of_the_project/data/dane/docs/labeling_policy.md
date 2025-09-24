Classes you’ll use later for training:
0 = background, 1 = healthy, 2 = bleached
(If you don’t truly have “dead” anywhere, write that you’ll train with 3 classes.)

Fusion rule for Type C (two separate masks → one combined mask):

Pixels in bleached mask → class 2

Pixels in non-bleached mask → class 1

If both overlap (rare), prefer bleached (2) and note as an edge case

Elsewhere → background (0)

Image-level labels (Type B): exact list of allowed labels and spelling (e.g., “bleached”, not “Bleach”).

Quality flags: define “gold / ok / unsure”.