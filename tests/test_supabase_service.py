from app.services.supabase_service import SupabaseService


def test_connection_terminated_protocol_error_is_transient():
    error = RuntimeError(
        "<ConnectionTerminated error_code:ErrorCodes.PROTOCOL_ERROR, "
        "last_stream_id:69, additional_data:None>"
    )

    assert SupabaseService._is_transient_connection_error(error) is True


def test_resource_temporarily_unavailable_is_transient():
    error = OSError(11, "Resource temporarily unavailable")

    assert SupabaseService._is_transient_connection_error(error) is True


def test_postgres_statement_timeout_is_transient():
    error = RuntimeError(
        "{'code': '57014', 'message': 'canceling statement due to statement timeout'}"
    )

    assert SupabaseService._is_transient_connection_error(error) is True
