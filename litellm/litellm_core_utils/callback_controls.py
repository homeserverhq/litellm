from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.constants import X_LITELLM_DISABLE_CALLBACKS
from litellm.integrations.custom_logger import CustomLogger

_DISABLED_CALLBACKS_KEY: Final = "litellm_disabled_callbacks"
_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_CALLBACK_NAMES_ADAPTER: Final = TypeAdapter(list[str])
_EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})


class CallbackControls:
    @staticmethod
    def get_disabled_callbacks(
        litellm_params: Mapping[str, object] | None,
        standard_callback_dynamic_params: Mapping[str, object] | None,
    ) -> frozenset[str] | None:
        proxy_request_value: Final = litellm_params.get("proxy_server_request") if litellm_params is not None else None
        proxy_request: Final[Mapping[str, object]] = (
            _MAPPING_ADAPTER.validate_python(proxy_request_value)
            if isinstance(proxy_request_value, Mapping)
            else _EMPTY_MAPPING
        )
        headers_value: Final = proxy_request.get("headers")
        request_headers: Final[Mapping[str, object]] = (
            _MAPPING_ADAPTER.validate_python(headers_value) if headers_value is not None else _EMPTY_MAPPING
        )
        for header_name, header_value in request_headers.items():
            if header_name.lower() != X_LITELLM_DISABLE_CALLBACKS:
                continue
            if isinstance(header_value, str):
                return frozenset(name.strip().lower() for name in header_value.split(",") if name.strip())

        disabled_value: Final = (standard_callback_dynamic_params or {}).get(_DISABLED_CALLBACKS_KEY)
        if isinstance(disabled_value, str):
            return frozenset(name.strip().lower() for name in disabled_value.split(",") if name.strip())
        if isinstance(disabled_value, (list, tuple)):
            names: Final = _CALLBACK_NAMES_ADAPTER.validate_python(disabled_value)
            return frozenset(name.lower() for name in names if name)
        return None

    @staticmethod
    def dynamic_disabling_allowed() -> bool:
        if litellm.allow_dynamic_callback_disabling is not True:
            return False
        try:
            from litellm.proxy.proxy_server import premium_user
        except ImportError:
            return False
        if premium_user:
            return True
        verbose_logger.debug("Dynamic callback disabling requires a premium user")
        return False

    @staticmethod
    def is_callback_disabled_dynamically(
        callback: litellm.CALLBACK_TYPES,
        litellm_params: Mapping[str, object] | None,
        standard_callback_dynamic_params: Mapping[str, object] | None,
    ) -> bool:
        try:
            result: Final = CallbackControls._is_callback_disabled_dynamically(
                callback,
                litellm_params,
                standard_callback_dynamic_params,
            )
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            verbose_logger.debug("Error checking disabled callbacks header: %s", e)
            return False
        return result

    @staticmethod
    def _is_callback_disabled_dynamically(
        callback: litellm.CALLBACK_TYPES,
        litellm_params: Mapping[str, object] | None,
        standard_callback_dynamic_params: Mapping[str, object] | None,
    ) -> bool:
        disabled_callbacks: Final = CallbackControls.get_disabled_callbacks(
            litellm_params,
            standard_callback_dynamic_params,
        )
        if not disabled_callbacks or not CallbackControls.dynamic_disabling_allowed():
            return False

        if isinstance(callback, str):
            return callback.lower() in disabled_callbacks

        if isinstance(callback, CustomLogger):
            from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry

            callback_name: Final = CustomLoggerRegistry.get_callback_str_from_class_type(callback.__class__)
            return callback_name is not None and callback_name.lower() in disabled_callbacks

        return False
