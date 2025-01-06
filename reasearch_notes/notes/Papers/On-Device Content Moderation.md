# On-Device Content Moderation
website: https://arxiv.org/abs/2107.11845

## Claims
F1 score = 0.91
precision = 95%
recall = 88%
false positive rate on safe images = 0.002

## Techniques 
### What exists:
- Skin detection based
- Hand crafted feature based: Bag-of-Visual-Words (BoVW)
- Neural feature based

### Solution proposed
- Bodypart Detector: Single Shot Multibox Detector (SSD) 
- MobileNetV3
Here a trade-off is made, indeed the aim is to run the model on mobile devices

## Overall
Image moderation, only safe and not safe for work (nsfw). 
Neither the dataset not the model or code are provided. Only comparison is OpenYahoo.
Great performance if we believe reported metrics.

