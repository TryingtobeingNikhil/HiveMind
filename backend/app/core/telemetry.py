"""
app/core/telemetry.py
──────────────────────
OpenTelemetry configuration for tracing.
"""

from __future__ import annotations

import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from app.core.config import Settings

logger = logging.getLogger(__name__)

def configure_telemetry(settings: Settings) -> None:
    if not settings.otel_enabled:
        # Use no-op tracer
        trace.set_tracer_provider(trace.NoOpTracerProvider())
        return

    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    logger.info("OpenTelemetry tracing configured", extra={
        "endpoint": settings.otel_endpoint,
        "service_name": settings.otel_service_name,
    })

def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
