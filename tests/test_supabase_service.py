from app.services.supabase_service import SupabaseService


def test_connection_terminated_protocol_error_is_transient():
    error = RuntimeError(
        "<ConnectionTerminated error_code:ErrorCodes.PROTOCOL_ERROR, "
        "last_stream_id:69, additional_data:None>"
    )

    assert SupabaseService._is_transient_connection_error(error) is True
