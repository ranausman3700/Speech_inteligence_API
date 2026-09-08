# syntax=docker/dockerfile:1.7

FROM python:3.11.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    HF_XET_CACHE=/models/huggingface/xet \
    XDG_CACHE_HOME=/models/cache

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /models /var/lib/speech-intelligence \
    && chown -R app:app /models /var/lib/speech-intelligence

# Fonts PDF export embeds. The CJK face is deliberately excluded: it alone is
# roughly twenty times this whole set, so install it only where callers need it.
ADD --chown=app:app \
    https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf \
    /fonts/NotoSans-Regular.ttf
ADD --chown=app:app \
    https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansArabic/hinted/ttf/NotoSansArabic-Regular.ttf \
    /fonts/NotoSansArabic-Regular.ttf
ADD --chown=app:app \
    https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansHebrew/hinted/ttf/NotoSansHebrew-Regular.ttf \
    /fonts/NotoSansHebrew-Regular.ttf
ADD --chown=app:app \
    https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf \
    /fonts/NotoSansDevanagari-Regular.ttf
ADD --chown=app:app \
    https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansBengali/hinted/ttf/NotoSansBengali-Regular.ttf \
    /fonts/NotoSansBengali-Regular.ttf
ADD --chown=app:app \
    https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansThai/hinted/ttf/NotoSansThai-Regular.ttf \
    /fonts/NotoSansThai-Regular.ttf

WORKDIR /app

COPY pyproject.toml ./

RUN python -c "import subprocess,sys,tomllib; data=tomllib.load(open('pyproject.toml','rb')); subprocess.check_call([sys.executable,'-m','pip','install',*data['build-system']['requires'],*data['project']['dependencies'],*data['project']['optional-dependencies']['export']])"

COPY README.md ./
COPY src ./src

RUN python -m pip install --no-deps --no-build-isolation .

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]

CMD ["uvicorn", "speech_intelligence_api.entrypoints.http.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=127.0.0.1"]
