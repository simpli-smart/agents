import asyncio
import os
import traceback

import aiohttp
from pydantic import BaseModel

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger
from .models import TTSModels

SIMPLISMART_BASE_URL = "https://api.simplismart.live/tts"
QWEN_BASE_URL = "https://api.simplismart.live/v1/audio/speech"
DEFAULT_ORPHEUS_MODEL = "canopylabs/orpheus-3b-0.1-ft"
DEFAULT_QWEN_MODEL = "qwen-tts"
DEFAULT_ORPHEUS_VOICE = "tara"
DEFAULT_QWEN_VOICE = "Chelsie"


class SimplismartTTSOptions(BaseModel):
    """Configuration options for SimpliSmart TTS models."""

    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.5
    max_tokens: int = 1000


class QwenTTSOptions(BaseModel):
    """Configuration options for Qwen 3 TTS."""

    language: str = "English"
    leading_silence: bool = True


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: TTSModels | str | None = None,
        voice: str | None = None,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        # sample_rate controls how the framework decodes/plays back the returned PCM audio;
        # it is not sent to the server.
        sample_rate: int = 24000,
        options: SimplismartTTSOptions | QwenTTSOptions | None = None,
        # Legacy Orpheus keyword args kept for backwards compatibility.
        # Ignored when options=QwenTTSOptions(...) is passed.
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.5,
        max_tokens: int = 1000,
    ) -> None:
        """
        Configuration options for SimpliSmart TTS (Text-to-Speech) models.

        Supports both Orpheus and Qwen 3 TTS models. Pass ``options=QwenTTSOptions(...)``
        to use the Qwen 3 endpoint; omit ``options`` (or pass ``SimplismartTTSOptions``)
        for the legacy Orpheus endpoint. All defaults for ``base_url``, ``model``, and
        ``voice`` are auto-detected from the ``options`` type.

        Args:
            base_url: Base URL for the TTS endpoint. Auto-detected from ``options`` type if not provided.
            model: TTS model to use. Auto-detected from ``options`` type if not provided.
            voice: Voice/speaker identifier. Auto-detected from ``options`` type if not provided.
            api_key: API key for authentication (defaults to SIMPLISMART_API_KEY env var).
            http_session: Optional aiohttp session for reuse.
            sample_rate: Expected sample rate of the returned PCM audio (default: 24000).
                Not sent to the server; used by the framework for playback.
            options: Pass ``QwenTTSOptions`` for Qwen 3 or ``SimplismartTTSOptions`` for
                Orpheus. Defaults to ``SimplismartTTSOptions`` for backwards compatibility.
            temperature: Orpheus only. Ignored for Qwen 3.
            top_p: Orpheus only. Ignored for Qwen 3.
            repetition_penalty: Orpheus only. Ignored for Qwen 3.
            max_tokens: Orpheus only. Ignored for Qwen 3.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        # Resolve options: explicit `options` kwarg takes precedence; fall back to building
        # SimplismartTTSOptions from the legacy flat kwargs for backwards compatibility.
        self._opts: SimplismartTTSOptions | QwenTTSOptions
        if options is None:
            self._opts = SimplismartTTSOptions(
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens,
            )
        else:
            self._opts = options

        # Auto-detect base_url, model, and voice from the options type so callers don't
        # need to specify them when switching between Orpheus and Qwen 3.
        if isinstance(self._opts, QwenTTSOptions):
            self._base_url = base_url if base_url is not None else QWEN_BASE_URL
            self._model = model if model is not None else DEFAULT_QWEN_MODEL
            self._voice = voice if voice is not None else DEFAULT_QWEN_VOICE
        else:
            self._base_url = base_url if base_url is not None else SIMPLISMART_BASE_URL
            self._model = model if model is not None else DEFAULT_ORPHEUS_MODEL
            self._voice = voice if voice is not None else DEFAULT_ORPHEUS_VOICE

        self._api_key = api_key or os.environ.get("SIMPLISMART_API_KEY")
        if not self._api_key:
            raise ValueError("SIMPLISMART_API_KEY is not set")

        self._session = http_session

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "SimpliSmart"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        """Synthesize text to speech.

        Args:
            text: Text to synthesize.
            conn_options: Connection options for the API request.

        Returns:
            ChunkedStream: Stream of audio data.
        """
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = tts._opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the TTS synthesis and stream audio chunks.

        Uses type-based detection to determine which API format to use:
        - QwenTTSOptions → Qwen 3 endpoint format
        - SimplismartTTSOptions → Orpheus endpoint format
        """
        logger.debug(
            f"TTS synthesis starting - text: {self._input_text[:50]}... voice: {self._tts._voice}"
        )

        # Determine payload format based on options type
        if isinstance(self._opts, QwenTTSOptions):
            # Qwen 3 TTS format
            payload = {
                "text": self._input_text,
                "language": self._opts.language,
                "speaker": self._tts._voice,
                "leading_silence": self._opts.leading_silence,
            }
            headers = {
                "Authorization": f"Bearer {self._tts._api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/L16",
            }
        else:
            # Orpheus TTS format
            payload = self._opts.model_dump()
            payload["prompt"] = self._input_text
            payload["voice"] = self._tts._voice
            payload["model"] = self._tts._model
            headers = {
                "Authorization": f"Bearer {self._tts._api_key}",
                "Content-Type": "application/json",
            }

        logger.debug(
            f"TTS request to {self._tts._base_url} with payload type: {type(self._opts).__name__}"
        )

        try:
            async with self._tts._ensure_session().post(
                self._tts._base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                logger.debug(f"TTS response status: {resp.status}")
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Simplismart TTS API error: {resp.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Simplismart TTS API Error: {error_text}",
                        status_code=resp.status,
                        request_id=None,
                        body=error_text,
                    )

                # Initialize audio emitter
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type="audio/pcm",
                )
                logger.debug(
                    f"TTS audio emitter initialized - sample_rate: {self._tts.sample_rate}"
                )

                # Stream audio chunks
                chunk_count = 0
                total_bytes = 0
                async for audio_data, _ in resp.content.iter_chunks():
                    if audio_data:
                        chunk_count += 1
                        total_bytes += len(audio_data)
                        output_emitter.push(audio_data)

                logger.debug(f"TTS received {chunk_count} chunks, {total_bytes} bytes total")
                output_emitter.flush()
                logger.debug("TTS synthesis completed successfully")

        except asyncio.TimeoutError as e:
            logger.error(f"Simplismart TTS API timeout: {e}")
            raise APITimeoutError("Simplismart TTS API request timed out") from e
        except aiohttp.ClientError as e:
            logger.error(f"Simplismart TTS API client error: {e}")
            raise APIConnectionError(f"Simplismart TTS API connection error: {e}") from e
        except APIStatusError:
            raise
        except Exception as e:
            logger.error(f"Error during Simplismart TTS processing: {traceback.format_exc()}")
            raise APIConnectionError(f"Unexpected error in Simplismart TTS: {e}") from e
