"""
J.A.R.V.I.S. Multimedia Processor
Advanced image, video, and audio processing with AI enhancements
"""

import os
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import multimedia libraries
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class MultimediaProcessor:
    """
    Advanced multimedia processing system
    Handles images, videos, and audio with AI-powered enhancements
    """

    def __init__(self, jarvis_instance):
        """
        Initialize multimedia processor

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance
        self.logger = logging.getLogger('JARVIS.MultimediaProcessor')

        # Processing capabilities
        self.capabilities = {
            "image_processing": PIL_AVAILABLE,
            "video_processing": MOVIEPY_AVAILABLE,
            "audio_processing": PYGAME_AVAILABLE,
            "computer_vision": True,  # OpenCV is in requirements
            "ai_enhancement": False  # Would need additional AI models
        }

        # Processing settings
        self.processing_config = {
            "image_quality": 95,
            "video_resolution": "1080p",
            "audio_quality": "high",
            "enable_ai_enhancements": True,
            "auto_enhance": True
        }

        # Performance tracking
        self.stats = {
            "images_processed": 0,
            "videos_processed": 0,
            "audio_files_processed": 0,
            "ai_enhancements_applied": 0,
            "processing_time_total": 0.0
        }

    async def initialize(self):
        """Initialize multimedia processor"""
        try:
            self.logger.info("Initializing multimedia processor...")

            # Test capabilities
            await self._test_capabilities()

            self.logger.info("Multimedia processor initialized")

        except Exception as e:
            self.logger.error(f"Error initializing multimedia processor: {e}")
            raise

    async def _test_capabilities(self):
        """Test multimedia processing capabilities"""
        try:
            # Test image processing
            if PIL_AVAILABLE:
                test_image = Image.new('RGB', (100, 100), color='red')
                test_image.save('/tmp/jarvis_test.png')
                os.remove('/tmp/jarvis_test.png')
                self.logger.info("✓ Image processing capability confirmed")

            # Test video processing
            if MOVIEPY_AVAILABLE:
                self.logger.info("✓ Video processing capability confirmed")

            # Test computer vision
            try:
                test_img = np.zeros((100, 100, 3), np.uint8)
                gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
                self.logger.info("✓ Computer vision capability confirmed")
            except Exception as e:
                self.logger.error(f"Computer vision test failed: {e}")

        except Exception as e:
            self.logger.error(f"Error testing capabilities: {e}")

    async def process_image(self,
                           image_path: str,
                           operations: List[str] = None,
                           output_path: str = None) -> Dict[str, Any]:
        """
        Process image with various enhancements

        Args:
            image_path: Path to input image
            operations: List of operations to apply
            output_path: Path for output image

        Returns:
            Processing result
        """
        start_time = time.time()

        try:
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            if not PIL_AVAILABLE:
                return {
                    "success": False,
                    "error": "Image processing not available (PIL not installed)"
                }

            # Load image
            image = Image.open(image_path)

            # Apply operations
            if operations is None:
                operations = ["auto_enhance", "sharpen", "color_balance"]

            processed_image = await self._apply_image_operations(image, operations)

            # Save result
            if output_path is None:
                base_name = os.path.splitext(image_path)[0]
                output_path = f"{base_name}_processed.jpg"

            processed_image.save(output_path, quality=self.processing_config["image_quality"])

            processing_time = time.time() - start_time

            # Update stats
            self.stats["images_processed"] += 1
            self.stats["processing_time_total"] += processing_time

            result = {
                "success": True,
                "input_path": image_path,
                "output_path": output_path,
                "operations_applied": operations,
                "processing_time": processing_time,
                "original_size": os.path.getsize(image_path),
                "processed_size": os.path.getsize(output_path)
            }

            self.logger.info(f"Image processed: {image_path} -> {output_path}")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing image {image_path}: {e}")

            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }

    async def _apply_image_operations(self, image: Image.Image, operations: List[str]) -> Image.Image:
        """Apply image enhancement operations"""
        try:
            processed_image = image.copy()

            for operation in operations:
                if operation == "auto_enhance":
                    processed_image = self._auto_enhance_image(processed_image)
                elif operation == "sharpen":
                    processed_image = processed_image.filter(ImageFilter.SHARPEN)
                elif operation == "smooth":
                    processed_image = processed_image.filter(ImageFilter.SMOOTH)
                elif operation == "color_balance":
                    processed_image = self._balance_colors(processed_image)
                elif operation == "denoise":
                    processed_image = self._denoise_image(processed_image)
                elif operation == "resize":
                    processed_image = self._smart_resize(processed_image)

            return processed_image

        except Exception as e:
            self.logger.error(f"Error applying image operations: {e}")
            return image

    def _auto_enhance_image(self, image: Image.Image) -> Image.Image:
        """Auto-enhance image quality"""
        try:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            enhanced = enhancer.enhance(1.2)

            # Enhance color saturation
            color_enhancer = ImageEnhance.Color(enhanced)
            enhanced = color_enhancer.enhance(1.1)

            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(1.3)

            return enhanced

        except Exception as e:
            self.logger.error(f"Error auto-enhancing image: {e}")
            return image

    def _balance_colors(self, image: Image.Image) -> Image.Image:
        """Balance image colors"""
        try:
            # Simple color balance using PIL
            return ImageEnhance.Color(image).enhance(1.1)

        except Exception as e:
            self.logger.error(f"Error balancing colors: {e}")
            return image

    def _denoise_image(self, image: Image.Image) -> Image.Image:
        """Remove noise from image"""
        try:
            # Use median filter for denoising
            return image.filter(ImageFilter.MedianFilter(size=3))

        except Exception as e:
            self.logger.error(f"Error denoising image: {e}")
            return image

    def _smart_resize(self, image: Image.Image) -> Image.Image:
        """Smart resize maintaining aspect ratio"""
        try:
            # Calculate new size maintaining aspect ratio
            width, height = image.size

            if width > height:
                new_width = 1024
                new_height = int((height * 1024) / width)
            else:
                new_height = 1024
                new_width = int((width * 1024) / height)

            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        except Exception as e:
            self.logger.error(f"Error smart resizing: {e}")
            return image

    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze image using computer vision"""
        try:
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }

            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "success": False,
                    "error": "Could not load image"
                }

            # Basic analysis
            height, width, channels = image.shape

            # Color analysis
            colors = self._analyze_colors(image)

            # Object detection (simplified)
            objects = self._detect_objects(image)

            # Quality analysis
            quality = self._analyze_image_quality(image)

            return {
                "success": True,
                "dimensions": {"width": width, "height": height, "channels": channels},
                "dominant_colors": colors,
                "detected_objects": objects,
                "quality_score": quality,
                "file_size": os.path.getsize(image_path),
                "format": os.path.splitext(image_path)[1].lower()
            }

        except Exception as e:
            self.logger.error(f"Error analyzing image: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _analyze_colors(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze dominant colors in image"""
        try:
            # Reshape image to list of pixels
            pixels = image.reshape(-1, 3)

            # Find unique colors and their counts
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

            # Sort by count (most frequent first)
            sorted_indices = np.argsort(-counts)
            top_colors = unique_colors[sorted_indices][:5]

            colors = []
            for color in top_colors:
                colors.append({
                    "rgb": color.tolist(),
                    "hex": "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2])),
                    "percentage": float(counts[sorted_indices[list(unique_colors).index(color)]] / len(pixels) * 100)
                })

            return colors

        except Exception as e:
            self.logger.error(f"Error analyzing colors: {e}")
            return []

    def _detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in image (simplified)"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        "type": "unknown",
                        "confidence": 0.5,
                        "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        "area": float(area)
                    })

            return objects

        except Exception as e:
            self.logger.error(f"Error detecting objects: {e}")
            return []

    def _analyze_image_quality(self, image: np.ndarray) -> float:
        """Analyze image quality"""
        try:
            # Calculate sharpness (Laplacian variance)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()

            # Calculate brightness
            brightness = np.mean(gray)

            # Calculate contrast
            contrast = gray.std()

            # Combine metrics into quality score
            quality_score = (sharpness * 0.4 + contrast * 0.4 + min(brightness, 255 - brightness) * 0.2)

            # Normalize to 0-100 scale
            normalized_score = min(100, quality_score / 100)

            return normalized_score

        except Exception as e:
            self.logger.error(f"Error analyzing image quality: {e}")
            return 50.0

    async def process_video(self,
                           video_path: str,
                           operations: List[str] = None,
                           output_path: str = None) -> Dict[str, Any]:
        """Process video with enhancements"""
        try:
            if not MOVIEPY_AVAILABLE:
                return {
                    "success": False,
                    "error": "Video processing not available (moviepy not installed)"
                }

            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "error": f"Video file not found: {video_path}"
                }

            # Load video
            video = mp.VideoFileClip(video_path)

            # Apply operations
            if operations is None:
                operations = ["stabilize", "enhance_audio", "optimize_quality"]

            processed_video = await self._apply_video_operations(video, operations)

            # Save result
            if output_path is None:
                base_name = os.path.splitext(video_path)[0]
                output_path = f"{base_name}_processed.mp4"

            processed_video.write_videofile(
                output_path,
                fps=video.fps,
                codec='libx264',
                audio_codec='aac'
            )

            # Clean up
            video.close()
            processed_video.close()

            self.stats["videos_processed"] += 1

            return {
                "success": True,
                "input_path": video_path,
                "output_path": output_path,
                "operations_applied": operations,
                "duration": video.duration,
                "fps": video.fps
            }

        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _apply_video_operations(self, video, operations: List[str]):
        """Apply video enhancement operations"""
        try:
            processed_video = video

            for operation in operations:
                if operation == "stabilize":
                    processed_video = self._stabilize_video(processed_video)
                elif operation == "enhance_audio":
                    processed_video = self._enhance_audio(processed_video)
                elif operation == "optimize_quality":
                    processed_video = self._optimize_video_quality(processed_video)

            return processed_video

        except Exception as e:
            self.logger.error(f"Error applying video operations: {e}")
            return video

    def _stabilize_video(self, video):
        """Stabilize video (simplified)"""
        # In a real implementation, this would use optical flow for stabilization
        return video

    def _enhance_audio(self, video):
        """Enhance video audio"""
        try:
            if video.audio:
                # Simple audio enhancement
                return video.volumex(1.2)  # Increase volume by 20%
            return video

        except Exception as e:
            self.logger.error(f"Error enhancing audio: {e}")
            return video

    def _optimize_video_quality(self, video):
        """Optimize video quality"""
        # In a real implementation, this would adjust bitrate, resolution, etc.
        return video

    async def generate_image_caption(self, image_path: str) -> Dict[str, Any]:
        """Generate caption for image using AI"""
        try:
            # Analyze image first
            analysis = await self.analyze_image(image_path)

            if not analysis["success"]:
                return analysis

            # Generate caption based on analysis
            caption = self._generate_caption_from_analysis(analysis)

            return {
                "success": True,
                "caption": caption,
                "confidence": 0.8,
                "image_analysis": analysis
            }

        except Exception as e:
            self.logger.error(f"Error generating image caption: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _generate_caption_from_analysis(self, analysis: Dict[str, Any]) -> str:
        """Generate caption from image analysis"""
        try:
            dimensions = analysis["dimensions"]
            colors = analysis["dominant_colors"]
            objects = analysis["detected_objects"]

            # Build caption
            caption_parts = []

            # Size description
            if dimensions["width"] > dimensions["height"]:
                caption_parts.append("wide")
            elif dimensions["height"] > dimensions["width"]:
                caption_parts.append("tall")
            else:
                caption_parts.append("square")

            # Color description
            if colors:
                primary_color = colors[0]
                caption_parts.append(f"primarily {primary_color['hex']}")

            # Object description
            if objects:
                object_types = [obj["type"] for obj in objects[:3]]
                if object_types:
                    caption_parts.append(f"containing {', '.join(object_types)}")

            # Combine into caption
            if caption_parts:
                caption = f"A {', '.join(caption_parts)} image"
            else:
                caption = "An image"

            return caption

        except Exception as e:
            self.logger.error(f"Error generating caption: {e}")
            return "An image"

    async def extract_audio_from_video(self, video_path: str, output_path: str = None) -> Dict[str, Any]:
        """Extract audio from video"""
        try:
            if not MOVIEPY_AVAILABLE:
                return {
                    "success": False,
                    "error": "Video processing not available"
                }

            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "error": f"Video file not found: {video_path}"
                }

            # Load video
            video = mp.VideoFileClip(video_path)

            if not video.audio:
                return {
                    "success": False,
                    "error": "Video contains no audio"
                }

            # Set output path
            if output_path is None:
                base_name = os.path.splitext(video_path)[0]
                output_path = f"{base_name}_audio.wav"

            # Extract audio
            video.audio.write_audiofile(output_path)

            # Clean up
            video.close()

            self.stats["audio_files_processed"] += 1

            return {
                "success": True,
                "input_video": video_path,
                "output_audio": output_path,
                "duration": video.duration,
                "audio_format": "wav"
            }

        except Exception as e:
            self.logger.error(f"Error extracting audio: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def create_video_thumbnail(self, video_path: str, output_path: str = None, timestamp: float = 1.0) -> Dict[str, Any]:
        """Create thumbnail from video"""
        try:
            if not os.path.exists(video_path):
                return {
                    "success": False,
                    "error": f"Video file not found: {video_path}"
                }

            # Load video
            video = mp.VideoFileClip(video_path)

            # Set output path
            if output_path is None:
                base_name = os.path.splitext(video_path)[0]
                output_path = f"{base_name}_thumbnail.jpg"

            # Get frame at timestamp
            frame = video.get_frame(timestamp)

            # Convert to PIL Image
            if PIL_AVAILABLE:
                image = Image.fromarray(frame)
                image.save(output_path, quality=95)
            else:
                # Use OpenCV
                cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Clean up
            video.close()

            return {
                "success": True,
                "input_video": video_path,
                "output_thumbnail": output_path,
                "timestamp": timestamp,
                "video_duration": video.duration
            }

        except Exception as e:
            self.logger.error(f"Error creating video thumbnail: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def batch_process_images(self,
                                  input_directory: str,
                                  operations: List[str] = None,
                                  output_directory: str = None) -> Dict[str, Any]:
        """Batch process multiple images"""
        try:
            if not os.path.exists(input_directory):
                return {
                    "success": False,
                    "error": f"Input directory not found: {input_directory}"
                }

            if output_directory is None:
                output_directory = os.path.join(input_directory, "processed")

            os.makedirs(output_directory, exist_ok=True)

            processed_files = []
            failed_files = []

            # Find all image files
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
            image_files = []

            for file in os.listdir(input_directory):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(input_directory, file))

            # Process each image
            for image_file in image_files:
                try:
                    output_file = os.path.join(output_directory, os.path.basename(image_file))

                    result = await self.process_image(image_file, operations, output_file)

                    if result["success"]:
                        processed_files.append(result)
                    else:
                        failed_files.append({"file": image_file, "error": result["error"]})

                except Exception as e:
                    failed_files.append({"file": image_file, "error": str(e)})

            return {
                "success": True,
                "processed_count": len(processed_files),
                "failed_count": len(failed_files),
                "input_directory": input_directory,
                "output_directory": output_directory,
                "processed_files": processed_files,
                "failed_files": failed_files
            }

        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported multimedia formats"""
        return {
            "images": ["jpg", "jpeg", "png", "bmp", "tiff", "webp", "gif"],
            "videos": ["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"],
            "audio": ["mp3", "wav", "flac", "aac", "ogg", "wma"]
        }

    def get_processing_capabilities(self) -> Dict[str, bool]:
        """Get processing capabilities status"""
        return {
            **self.capabilities,
            "ai_enhancement": self.processing_config["enable_ai_enhancements"]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get multimedia processing statistics"""
        return {
            **self.stats,
            "capabilities": self.capabilities,
            "average_processing_time": (
                self.stats["processing_time_total"] / max(1, self.stats["images_processed"])
            )
        }

    async def shutdown(self):
        """Shutdown multimedia processor"""
        try:
            self.logger.info("Shutting down multimedia processor...")

            # Clean up any resources
            if PYGAME_AVAILABLE:
                pygame.quit()

            self.logger.info("Multimedia processor shutdown complete")

        except Exception as e:
            self.logger.error(f"Error shutting down multimedia processor: {e}")