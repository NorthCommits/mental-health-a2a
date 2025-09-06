"""
Input Processors for Primary Screening Agent

Handles multi-modal input processing including text, audio, documents, and images
for mental health assessment and screening.
"""

import json
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import openai
from transformers import pipeline
import librosa
import cv2
import numpy as np
from PIL import Image
import pypdf
from docx import Document
from pydantic import BaseModel, Field


class ProcessedInput(BaseModel):
    """Processed input data from various modalities"""
    input_type: str  # "text", "audio", "image", "document"
    content: str
    confidence_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=datetime.utcnow)


class TextProcessor:
    """Processes text input for mental health assessment"""
    
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    
    async def process_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> ProcessedInput:
        """
        Process text input for mental health screening
        
        Args:
            text: Input text to process
            context: Additional context for processing
            
        Returns:
            ProcessedInput with extracted information
        """
        try:
            # Basic sentiment analysis
            sentiment_result = self.sentiment_analyzer(text)[0]
            
            # Emotion analysis
            emotion_result = self.emotion_analyzer(text)[0]
            
            # Extract mental health keywords and phrases
            mental_health_indicators = await self._extract_mental_health_indicators(text)
            
            # Generate structured assessment data
            assessment_data = await self._generate_assessment_data(text, context)
            
            metadata = {
                "sentiment": sentiment_result,
                "emotion": emotion_result,
                "mental_health_indicators": mental_health_indicators,
                "assessment_data": assessment_data,
                "word_count": len(text.split()),
                "processing_method": "nlp_analysis"
            }
            
            return ProcessedInput(
                input_type="text",
                content=text,
                confidence_score=0.8,  # Base confidence for text processing
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error processing text: {e}")
            return ProcessedInput(
                input_type="text",
                content=text,
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
    
    async def _extract_mental_health_indicators(self, text: str) -> Dict[str, Any]:
        """Extract mental health indicators from text"""
        # Keywords for different mental health conditions
        depression_keywords = [
            "depressed", "sad", "hopeless", "worthless", "empty", "lonely",
            "tired", "exhausted", "sleep", "appetite", "concentration"
        ]
        
        anxiety_keywords = [
            "anxious", "worried", "nervous", "panic", "fear", "restless",
            "tense", "stressed", "overwhelmed", "racing thoughts"
        ]
        
        crisis_keywords = [
            "suicide", "kill myself", "end it all", "not worth living",
            "better off dead", "hurt myself", "self harm"
        ]
        
        # Count keyword occurrences
        text_lower = text.lower()
        depression_count = sum(1 for keyword in depression_keywords if keyword in text_lower)
        anxiety_count = sum(1 for keyword in anxiety_keywords if keyword in text_lower)
        crisis_count = sum(1 for keyword in crisis_keywords if keyword in text_lower)
        
        return {
            "depression_indicators": depression_count,
            "anxiety_indicators": anxiety_count,
            "crisis_indicators": crisis_count,
            "total_indicators": depression_count + anxiety_count + crisis_count
        }
    
    async def _generate_assessment_data(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate structured assessment data from text"""
        try:
            # Use OpenAI to extract structured assessment data
            prompt = f"""
            Analyze the following text for mental health assessment indicators and extract structured data:
            
            Text: "{text}"
            
            Extract the following information:
            1. Mood indicators (scale 1-10)
            2. Sleep quality (scale 1-10)
            3. Energy level (scale 1-10)
            4. Anxiety level (scale 1-10)
            5. Social functioning (scale 1-10)
            6. Any specific symptoms mentioned
            7. Risk factors for self-harm or suicide
            
            Return as JSON format.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            assessment_text = response.choices[0].message.content
            # Try to parse as JSON, fallback to text if parsing fails
            try:
                return json.loads(assessment_text)
            except json.JSONDecodeError:
                return {"raw_assessment": assessment_text}
                
        except Exception as e:
            print(f"Error generating assessment data: {e}")
            return {"error": str(e)}


class AudioProcessor:
    """Processes audio input for mental health assessment"""
    
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    async def process_audio(self, audio_data: bytes, sample_rate: int = 22050) -> ProcessedInput:
        """
        Process audio input for mental health screening
        
        Args:
            audio_data: Raw audio data
            sample_rate: Sample rate of the audio
            
        Returns:
            ProcessedInput with extracted information
        """
        try:
            # Convert audio to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Extract audio features
            features = await self._extract_audio_features(audio_array, sample_rate)
            
            # Transcribe audio to text
            transcribed_text = await self._transcribe_audio(audio_array, sample_rate)
            
            # Analyze transcribed text
            if transcribed_text:
                text_processor = TextProcessor("")  # Will need API key
                text_analysis = await text_processor.process_text(transcribed_text)
            else:
                text_analysis = None
            
            metadata = {
                "audio_features": features,
                "transcribed_text": transcribed_text,
                "text_analysis": text_analysis.metadata if text_analysis else None,
                "duration": len(audio_array) / sample_rate,
                "sample_rate": sample_rate
            }
            
            return ProcessedInput(
                input_type="audio",
                content=transcribed_text or "",
                confidence_score=features.get("confidence", 0.5),
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return ProcessedInput(
                input_type="audio",
                content="",
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
    
    async def _extract_audio_features(self, audio_array: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract features from audio for mental health assessment"""
        try:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sample_rate)[0]
            
            # Extract rhythm features
            tempo, beats = librosa.beat.beat_track(y=audio_array, sr=sample_rate)
            
            # Extract zero crossing rate (indicator of voice quality)
            zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
            
            # Calculate voice quality indicators
            voice_quality = {
                "pitch_variation": np.std(spectral_centroids),
                "energy_variation": np.std(mfccs[0]),  # First MFCC coefficient
                "rhythm_regularity": 1.0 / (np.std(np.diff(beats)) + 1e-6),
                "voice_stability": 1.0 / (np.std(zcr) + 1e-6)
            }
            
            # Determine confidence based on audio quality
            confidence = min(1.0, voice_quality["voice_stability"] * 0.5 + 0.5)
            
            return {
                "mfccs": mfccs.tolist(),
                "spectral_centroids": spectral_centroids.tolist(),
                "tempo": float(tempo),
                "voice_quality": voice_quality,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return {"error": str(e), "confidence": 0.0}
    
    async def _transcribe_audio(self, audio_array: np.ndarray, sample_rate: int) -> Optional[str]:
        """Transcribe audio to text using speech recognition"""
        try:
            # This would typically use a speech recognition service
            # For now, return a placeholder
            return "Audio transcription not implemented in this example"
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None


class ImageProcessor:
    """Processes image input for mood assessment and facial expression analysis"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    async def process_image(self, image_data: bytes) -> ProcessedInput:
        """
        Process image input for mood assessment
        
        Args:
            image_data: Raw image data
            
        Returns:
            ProcessedInput with extracted information
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = await self._detect_faces(cv_image)
            
            # Analyze facial expressions
            expression_analysis = await self._analyze_facial_expressions(cv_image, faces)
            
            # Extract mood indicators
            mood_indicators = await self._extract_mood_indicators(cv_image, faces)
            
            metadata = {
                "faces_detected": len(faces),
                "expression_analysis": expression_analysis,
                "mood_indicators": mood_indicators,
                "image_dimensions": image.size,
                "processing_method": "facial_expression_analysis"
            }
            
            return ProcessedInput(
                input_type="image",
                content=f"Image with {len(faces)} face(s) detected",
                confidence_score=expression_analysis.get("confidence", 0.5),
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return ProcessedInput(
                input_type="image",
                content="",
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
    
    async def _detect_faces(self, cv_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the image"""
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return [(x, y, w, h) for (x, y, w, h) in faces]
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    async def _analyze_facial_expressions(self, cv_image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Analyze facial expressions for mood indicators"""
        try:
            if not faces:
                return {"confidence": 0.0, "expressions": []}
            
            expressions = []
            for (x, y, w, h) in faces:
                # Extract face region
                face_region = cv_image[y:y+h, x:x+w]
                
                # Basic expression analysis (simplified)
                # In a real implementation, this would use a trained model
                expression = {
                    "face_id": len(expressions),
                    "coordinates": (x, y, w, h),
                    "mood_score": 0.5,  # Placeholder
                    "confidence": 0.6   # Placeholder
                }
                expressions.append(expression)
            
            return {
                "confidence": 0.6,
                "expressions": expressions,
                "overall_mood": "neutral"  # Placeholder
            }
            
        except Exception as e:
            print(f"Error analyzing facial expressions: {e}")
            return {"confidence": 0.0, "expressions": [], "error": str(e)}
    
    async def _extract_mood_indicators(self, cv_image: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
        """Extract mood indicators from the image"""
        try:
            # Basic mood indicators based on image analysis
            # In a real implementation, this would use advanced computer vision
            mood_indicators = {
                "brightness_level": np.mean(cv_image),
                "color_vibrancy": np.std(cv_image),
                "face_count": len(faces),
                "mood_estimate": "neutral"  # Placeholder
            }
            
            return mood_indicators
            
        except Exception as e:
            print(f"Error extracting mood indicators: {e}")
            return {"error": str(e)}


class DocumentProcessor:
    """Processes document input for mental health assessment"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    async def process_document(self, document_data: bytes, filename: str) -> ProcessedInput:
        """
        Process document input for mental health screening
        
        Args:
            document_data: Raw document data
            filename: Name of the document file
            
        Returns:
            ProcessedInput with extracted information
        """
        try:
            # Determine file type
            file_extension = filename.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                content = await self._extract_pdf_content(document_data)
            elif file_extension == 'docx':
                content = await self._extract_docx_content(document_data)
            elif file_extension == 'txt':
                content = document_data.decode('utf-8')
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Analyze document content
            analysis = await self._analyze_document_content(content)
            
            metadata = {
                "filename": filename,
                "file_type": file_extension,
                "word_count": len(content.split()),
                "analysis": analysis,
                "processing_method": "document_analysis"
            }
            
            return ProcessedInput(
                input_type="document",
                content=content,
                confidence_score=analysis.get("confidence", 0.7),
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error processing document: {e}")
            return ProcessedInput(
                input_type="document",
                content="",
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
    
    async def _extract_pdf_content(self, pdf_data: bytes) -> str:
        """Extract text content from PDF"""
        try:
            pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_data))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting PDF content: {e}")
            return ""
    
    async def _extract_docx_content(self, docx_data: bytes) -> str:
        """Extract text content from DOCX"""
        try:
            doc = Document(io.BytesIO(docx_data))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting DOCX content: {e}")
            return ""
    
    async def _analyze_document_content(self, content: str) -> Dict[str, Any]:
        """Analyze document content for mental health indicators"""
        try:
            # Use text processor to analyze content
            text_processor = TextProcessor("")  # Will need API key
            processed_input = await text_processor.process_text(content)
            
            return {
                "confidence": processed_input.confidence_score,
                "mental_health_indicators": processed_input.metadata.get("mental_health_indicators", {}),
                "sentiment": processed_input.metadata.get("sentiment", {}),
                "emotion": processed_input.metadata.get("emotion", {})
            }
            
        except Exception as e:
            print(f"Error analyzing document content: {e}")
            return {"error": str(e), "confidence": 0.0}
