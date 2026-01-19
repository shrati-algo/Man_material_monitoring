# def raise_violation(cam_id,line_id,violation_type, message):
#     print(
#         f"🚨 VIOLATION | {cam_id} | {line_id} | {violation_type} | {message}"
#     )
#     # later:
#     # - write to DB
#     # - send API alert
#     # - trigger PLC

from database.db_manager import get_connection, init_violation_table

# Ensure table exists once
init_violation_table()

def raise_violation(cam_id, line_id, violation_type, message):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO man_material_violations
            (camera, line, violation_type, message)
            VALUES (%s, %s, %s, %s)
            """,
            (cam_id, line_id, violation_type, message)
        )

        print("insertion complete")
        cursor.close()
        conn.close()

    except Exception as e:
        print("❌ DB VIOLATION LOG ERROR:", e)
