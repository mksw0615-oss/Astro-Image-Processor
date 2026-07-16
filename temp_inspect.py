from PIL import Image, ImageStat, ImageFilter

img_path = r'C:\Users\seowo\OneDrive\Pictures\Engineering The Cosmos\Raw Saturn EXAMPLE.jpg'
img = Image.open(img_path).convert('RGB')
gray = img.convert('L')
stat = ImageStat.Stat(gray)

print('size', img.size)
print('avg', stat.mean[0])
print('std', stat.stddev[0])
print('brightest', max(gray.getextrema()))
print('sat', ImageStat.Stat(img.convert('HSV').split()[1]).mean[0])

edges = gray.filter(ImageFilter.FIND_EDGES)
print('edge', ImageStat.Stat(edges).mean[0])

width, height = gray.size
count = 0
min_x = width
min_y = height
max_x = 0
max_y = 0
for y in range(height):
    for x in range(width):
        value = gray.getpixel((x, y))
        if value >= 180:
            count += 1
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
if count == 0:
    score = 0.0
else:
    area_fraction = count / (width * height)
    bbox_width = max_x - min_x + 1
    bbox_height = max_y - min_y + 1
    bbox_area = bbox_width * bbox_height
    fill_ratio = count / bbox_area if bbox_area else 0.0
    aspect_ratio = min(bbox_width, bbox_height) / max(bbox_width, bbox_height) if max(bbox_width, bbox_height) else 0.0
    score = min(1.0, (area_fraction / 0.03) * 0.5 + fill_ratio * 0.3 + aspect_ratio * 0.2)
print('bright_object_score', score)

bbox = gray.getbbox()
if bbox:
    left, upper, right, lower = bbox
    bbox_width = right - left + 1
    bbox_height = lower - upper + 1
    bbox_area = bbox_width * bbox_height
    image_area = width * height
    compactness = bbox_area / image_area
    aspect_ratio = min(bbox_width, bbox_height) / max(bbox_width, bbox_height) if max(bbox_width, bbox_height) else 0.0
    print('point_like_score', min(1.0, compactness * 0.7 + aspect_ratio * 0.3))

sample = gray.resize((max(20, width // 10), max(20, height // 10)))
pixels = list(sample.getdata())
mean_brightness = sum(pixels) / len(pixels)
variance = sum((p - mean_brightness) ** 2 for p in pixels) / len(pixels)
print('diffuse_region_score', min(1.0, variance / 5000.0))
