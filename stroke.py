from strokes import strokes

name = '謝舒凱'
good = [1, 3, 5, 6, 7, 8, 11, 13, 15, 16, 17, 18, 21, 23, 24, 25, 29, 31, 32, 33, 35, 37, 39, 41, 45, 47, 48, 52, 63, 65, 67, 68, 81]
bad = [2, 4, 9, 10, 12, 14, 19, 20, 22, 28, 34, 36, 44, 46, 54, 56, 59, 60, 62, 64, 66, 69, 70, 74, 76, 79]
stroke_count = sum(strokes(name))
print(stroke_count)
if stroke_count in good:
    print('吉')
elif stroke_count in bad:
    print('兇')
else:
    print('半吉半兇')