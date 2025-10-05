def create_icon():
    """Create advanced JARVIS icon with multiple sizes and proper ICO format"""
    icon_path = Path('assets/jarvis.ico')
    icon_path.parent.mkdir(exist_ok=True)

    if not icon_path.exists():
        try:
            # Try to create a professional icon using PIL if available
            try:
                from PIL import Image, ImageDraw, ImageFont
                import io

                # Create multiple sizes for professional ICO file
                sizes = [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)]
                images = []

                for size in sizes:
                    # Create icon with JARVIS theme
                    img = Image.new('RGBA', size, (0, 0, 0, 0))
                    draw = ImageDraw.Draw(img)

                    # Scale elements based on size
                    scale = size[0] / 256
                    center = (size[0] // 2, size[1] // 2)
                    radius = int(100 * scale)

                    # Outer glow effect
                    for i in range(5):
                        alpha = max(10, 50 - i * 10)
                        glow_radius = radius + i
                        draw.ellipse(
                            [center[0] - glow_radius, center[1] - glow_radius,
                             center[0] + glow_radius, center[1] + glow_radius],
                            fill=(0, 150, 255, alpha)
                        )

                    # Main circle
                    draw.ellipse(
                        [center[0] - radius, center[1] - radius,
                         center[0] + radius, center[1] + radius],
                        fill=(0, 100, 200, 255),
                        outline=(0, 200, 255, 255),
                        width=max(1, int(3 * scale))
                    )

                    # Inner "J" design (scaled)
                    j_width = int(15 * scale)
                    j_height = int(60 * scale)
                    j_x = center[0] - int(8 * scale)
                    j_y = center[1] - int(30 * scale)

                    # Vertical line of J
                    draw.rectangle([j_x, j_y, j_x + j_width, j_y + j_height],
                                 fill=(255, 255, 255, 255))

                    # Top horizontal line
                    draw.rectangle([j_x - int(15 * scale), j_y,
                                  j_x + j_width, j_y + int(15 * scale)],
                                 fill=(255, 255, 255, 255))

                    # Bottom horizontal line
                    draw.rectangle([j_x - int(15 * scale), j_y + j_height - int(15 * scale),
                                  j_x, j_y + j_height],
                                 fill=(255, 255, 255, 255))

                    # Inner glow effects
                    for i in range(3):
                        glow_size = int(30 * scale) + i
                        draw.ellipse(
                            [center[0] - glow_size, center[1] - glow_size,
                             center[0] + glow_size, center[1] + glow_size],
                            fill=(255, 255, 255, max(5, 20 - i * 5))
                        )

                    images.append(img)

                # Save as ICO with multiple sizes
                images[0].save(str(icon_path), format='ICO', sizes=sizes, append_images=images[1:])
                print(f"✓ Created professional JARVIS icon with {len(sizes)} sizes: {icon_path}")

            except ImportError:
                # PIL not available, create advanced ICO programmatically
                print("PIL not available, creating advanced ICO programmatically...")
                ico_data = create_advanced_ico_programmatically()
                with open(icon_path, 'wb') as f:
                    f.write(ico_data)
                print(f"✓ Created advanced programmatic JARVIS icon: {icon_path}")

        except Exception as e:
            print(f"Warning: Could not create advanced icon: {e}. Creating basic fallback...")
            try:
                # Ultimate fallback - create a minimal working ICO
                ico_data = create_minimal_ico()
                with open(icon_path, 'wb') as f:
                    f.write(ico_data)
                print(f"✓ Created minimal fallback JARVIS icon: {icon_path}")
            except Exception as e2:
                print(f"Error: Could not create any icon: {e2}. Using system default.")
                return None

    return str(icon_path)


def create_advanced_ico_programmatically():
    """Create a professional ICO file programmatically without PIL"""
    # ICO Header
    ico_data = b'\x00\x00'  # Reserved
    ico_data += b'\x01\x00'  # Type (ICO)
    ico_data += b'\x05\x00'  # Number of images (5 sizes)

    # Image directory entries for 256x256, 128x128, 64x64, 32x32, 16x16
    sizes = [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)]
    image_data_list = []
    offset = 6 + len(sizes) * 16  # Header + directory entries

    for width, height in sizes:
        # Create BMP data for this size
        bmp_data = create_bmp_data(width, height)
        image_data_list.append(bmp_data)

        # Directory entry
        ico_data += bytes([width])  # Width
        ico_data += bytes([height])  # Height
        ico_data += b'\x00'  # Color count
        ico_data += b'\x00'  # Reserved
        ico_data += b'\x01\x00'  # Color planes
        ico_data += b'\x20\x00'  # Bits per pixel (32 for RGBA)

        # Image size
        size_bytes = len(bmp_data).to_bytes(4, byteorder='little')
        ico_data += size_bytes

        # Image offset
        offset_bytes = offset.to_bytes(4, byteorder='little')
        ico_data += offset_bytes

        offset += len(bmp_data)

    # Add actual image data
    for bmp_data in image_data_list:
        ico_data += bmp_data

    return ico_data


def create_bmp_data(width, height):
    """Create BMP image data for ICO format"""
    # BMP header for 32-bit RGBA
    # ICO uses BMP format internally

    # Calculate padding
    row_size = width * 4  # 4 bytes per pixel (RGBA)
    padding = (4 - (row_size % 4)) % 4

    # BMP header
    bmp_data = b'BM'  # Signature
    file_size = 40 + 14 + (row_size + padding) * height  # Header + pixel data
    bmp_data += file_size.to_bytes(4, byteorder='little')
    bmp_data += b'\x00\x00\x00\x00'  # Reserved
    bmp_data += (40 + 14).to_bytes(4, byteorder='little')  # Data offset

    # DIB header (BITMAPINFOHEADER)
    bmp_data += b'\x28\x00\x00\x00'  # Header size
    bmp_data += width.to_bytes(4, byteorder='little')
    bmp_data += height.to_bytes(4, byteorder='little')  # Note: ICO uses bottom-up
    bmp_data += b'\x01\x00'  # Planes
    bmp_data += b'\x20\x00'  # Bits per pixel (32)
    bmp_data += b'\x00\x00\x00\x00'  # Compression (BI_RGB)
    bmp_data += b'\x00\x00\x00\x00'  # Image size (can be 0 for BI_RGB)
    bmp_data += b'\x00\x00\x00\x00'  # X pixels per meter
    bmp_data += b'\x00\x00\x00\x00'  # Y pixels per meter
    bmp_data += b'\x00\x00\x00\x00'  # Colors used
    bmp_data += b'\x00\x00\x00\x00'  # Important colors

    # Generate pixel data (RGBA, bottom-up)
    for y in range(height - 1, -1, -1):  # Bottom to top
        for x in range(width):
            # Create JARVIS-inspired design
            center_x, center_y = width // 2, height // 2
            dx, dy = x - center_x, y - center_y
            distance = (dx * dx + dy * dy) ** 0.5

            # Circle with glow effect
            radius = min(width, height) * 0.3
            if distance <= radius:
                # Inside circle - JARVIS blue
                r, g, b, a = 0, 100, 200, 255
            elif distance <= radius + 10:
                # Glow effect
                alpha = max(0, int(255 * (1 - (distance - radius) / 10)))
                r, g, b, a = 0, 150, 255, alpha
            else:
                # Transparent background
                r, g, b, a = 0, 0, 0, 0

            # Add "J" shape in center
            if abs(dx) < 5 and abs(dy) < 15:  # Vertical line
                r, g, b, a = 255, 255, 255, 255
            elif dx >= -10 and dx < 5 and abs(dy) < 5:  # Top horizontal
                r, g, b, a = 255, 255, 255, 255
            elif dx >= -10 and dx < 0 and abs(dy - 10) < 5:  # Bottom horizontal
                r, g, b, a = 255, 255, 255, 255

            # BMP uses BGRA order
            bmp_data += bytes([b, g, r, a])

        # Add padding
        bmp_data += b'\x00' * padding

    return bmp_data


def create_minimal_ico():
    """Create a minimal working ICO file as ultimate fallback"""
    # ICO Header
    ico_data = b'\x00\x00'  # Reserved
    ico_data += b'\x01\x00'  # Type (ICO)
    ico_data += b'\x01\x00'  # Number of images

    # Image directory (16x16 icon)
    ico_data += b'\x10'  # Width
    ico_data += b'\x10'  # Height
    ico_data += b'\x00'  # Color count
    ico_data += b'\x00'  # Reserved
    ico_data += b'\x01\x00'  # Color planes
    ico_data += b'\x20\x00'  # Bits per pixel (32)

    # Size and offset
    image_size = 40 + 14 + 16 * 16 * 4  # BMP header + pixel data
    ico_data += image_size.to_bytes(4, byteorder='little')
    ico_data += b'\x16\x00\x00\x00'  # Offset to image data

    # BMP data for 16x16 RGBA icon
    bmp_data = b'BM'  # Signature
    bmp_data += image_size.to_bytes(4, byteorder='little')
    bmp_data += b'\x00\x00\x00\x00'  # Reserved
    bmp_data += b'\x36\x00\x00\x00'  # Data offset

    # DIB header
    bmp_data += b'\x28\x00\x00\x00'  # Header size
    bmp_data += b'\x10\x00\x00\x00'  # Width
    bmp_data += b'\x10\x00\x00\x00'  # Height
    bmp_data += b'\x01\x00'  # Planes
    bmp_data += b'\x20\x00'  # Bits per pixel (32)
    bmp_data += b'\x00\x00\x00\x00'  # Compression
    bmp_data += b'\x00\x00\x00\x00'  # Image size
    bmp_data += b'\x00\x00\x00\x00'  # X pixels per meter
    bmp_data += b'\x00\x00\x00\x00'  # Y pixels per meter
    bmp_data += b'\x00\x00\x00\x00'  # Colors used
    bmp_data += b'\x00\x00\x00\x00'  # Important colors

    # Simple 16x16 pixel data (BGRA, bottom-up)
    # Create a simple blue circle on transparent background
    for y in range(15, -1, -1):
        for x in range(16):
            center_x, center_y = 8, 8
            dx, dy = x - center_x, y - center_y
            distance = (dx * dx + dy * dy) ** 0.5

            if distance <= 6:
                # Blue circle
                b, g, r, a = 200, 100, 0, 255
            else:
                # Transparent
                b, g, r, a = 0, 0, 0, 0

            bmp_data += bytes([b, g, r, a])

    ico_data += bmp_data
    return ico_data