import json, os
import datetime
import time
import uuid

DEFAULT_COUNTRY_CODE = '+213'

DATA_DIR = "data"

def _path(name):
    return os.path.join(DATA_DIR, f"{name}.json")


def _ensure_file(name, default):
    p = _path(name)
    if not os.path.exists(p):
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
def load(name):
    with open(_path(name), "r", encoding="utf-8") as f:
        return json.load(f)

def save(name, data):
    with open(_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
def get_teachers():
    return load("teachers")

def set_current_teacher(name, subject):
    state = load("class_state")
    state["current_teacher"] = name
    state["subject"] = subject
    save("class_state", state)

def logout_teacher():
    state = load("class_state")
    state["current_teacher"] = None
    state["subject"] = None
    save("class_state", state)
def get_students():
    return load("students")

def mark_attendance(name, present):
    students = load("students")
    for s in students:
        if s["name"] == name:
            s["present"] = present
    save("students", students)


def _normalize_phone(phone: str) -> str:
    if not phone:
        return phone
    phone = phone.strip()
    if phone.startswith('00'):
        phone = '+' + phone[2:]
    # If the number starts with a single leading zero (local format, e.g. 077...)
    # convert to DEFAULT_COUNTRY_CODE + rest (e.g. +213774...)
    if phone.startswith('0') and not phone.startswith('00'):
        # remove leading zero and prepend default country
        phone = DEFAULT_COUNTRY_CODE + phone[1:]
    elif not phone.startswith('+'):
        phone = '+' + phone
    return phone

def get_class_state():
    return load("class_state")

def update_fire(fire, section=None):
    """Update class_state with fire status. Optionally include section name
    where the fire was detected so admin can be notified which class.
    """
    # ensure files exist
    _ensure_file('class_state', {"status": "SAFE", "fire_section": None, "current_teacher": None, "subject": None})
    _ensure_file('notifications', [])

    state = load("class_state")
    prev = state.get('status')
    state["status"] = "FIRE" if fire else "SAFE"
    state["fire_section"] = section if fire else None
    save("class_state", state)

    # log notifications for fire start/stop
    notes = load('notifications')
    now = datetime.datetime.now().isoformat()
    if prev != 'FIRE' and state['status'] == 'FIRE':
        # new fire event
        notes.append({
            'id': str(uuid.uuid4()),
            'type': 'fire',
            'section': section,
            'start_time': now,
            'end_time': None
        })
        save('notifications', notes)
    elif prev == 'FIRE' and state['status'] != 'FIRE':
        # close last fire event for this section
        for n in reversed(notes):
            if n.get('type') == 'fire' and n.get('section') == section and n.get('end_time') is None:
                n['end_time'] = now
                break
        save('notifications', notes)



def send_sms(phone, message):
    """Send message via WhatsApp using webbrowser.
    
    Opens a WhatsApp Web link in the default browser.
    The function logs every attempt in ``data/sent_messages.json`` and
    updates the ``status`` field to ``sending``, ``sent`` or ``failed``.
    """
    import urllib.parse
    import webbrowser

    # Normalize phone number
    p = _normalize_phone(phone)

    # Record message to log
    try:
        _ensure_file('sent_messages', [])
        msgs = load('sent_messages')
    except Exception:
        msgs = []

    entry = {
        'to': phone,
        'message': message,
        'timestamp': datetime.datetime.now().isoformat(),
        'status': 'sending'
    }
    msgs.append(entry)

    try:
        save('sent_messages', msgs)
        print(f"[SMS/LOG] Preparing to send message to {p}")
    except Exception as e:
        print(f"[ERROR] Failed to record message: {e}")

    def _mark(status):
        try:
            msgs_upd = load('sent_messages')
            msgs_upd[-1]['status'] = status
            save('sent_messages', msgs_upd)
        except Exception:
            pass

    # Open WhatsApp Web link in default browser
    try:
        encoded_msg = urllib.parse.quote(message)
        whatsapp_url = f"https://wa.me/{p[1:]}?text={encoded_msg}"
        print(f"[WHATSAPP] Opening: {whatsapp_url}")
        webbrowser.open(whatsapp_url)
        print(f"[SUCCESS] WhatsApp opened for {p}")
        _mark('sent')
    except Exception as e:
        print(f"[ERROR] Failed to open WhatsApp: {e}")
        _mark('failed')


def schedule_absent_notifications(student_entries, delay_seconds=600):
    """student_entries: list of dicts with keys 'name' and 'parent_phone' and optional 'section'"""
    _ensure_file('delayed_notifications', [])
    pending = load('delayed_notifications')
    ts = (datetime.datetime.now() + datetime.timedelta(seconds=delay_seconds)).isoformat()
    for s in student_entries:
        pending.append({
            'id': str(uuid.uuid4()),
            'name': s.get('name'),
            'parent_phone': s.get('parent_phone'),
            'section': s.get('section'),
            'message': s.get('message'),
            'scheduled_time': ts
        })
    save('delayed_notifications', pending)


def cancel_scheduled_for_student(name):
    _ensure_file('delayed_notifications', [])
    pending = load('delayed_notifications')
    pending = [p for p in pending if p.get('name') != name]
    save('delayed_notifications', pending)


def process_scheduled_notifications():
    """Send any scheduled notifications that are due. Should be called periodically by apps."""
    _ensure_file('delayed_notifications', [])
    pending = load('delayed_notifications')
    now = datetime.datetime.now()
    remaining = []
    for p in pending:
        try:
            sched = datetime.datetime.fromisoformat(p.get('scheduled_time'))
        except Exception:
            remaining.append(p)
            continue
        if sched <= now:
            phone = p.get('parent_phone')
            msg = p.get('message') or f"Student {p.get('name')} is absent today in section {p.get('section')}"
            try:
                send_sms(phone, msg)
            except Exception as e:
                print('[SEND ERROR]', e)
        else:
            remaining.append(p)
    save('delayed_notifications', remaining)


def get_notifications():
    _ensure_file('notifications', [])
    return load('notifications')


def clear_notification(nid):
    _ensure_file('notifications', [])
    notes = load('notifications')
    notes = [n for n in notes if n.get('id') != nid]
    save('notifications', notes)


def get_sent_messages():
    _ensure_file('sent_messages', [])
    return load('sent_messages')


# ---------- HEARTBEAT / PRESENCE HELPERS ----------
def touch_heartbeat(teacher_name=None):
    """Update class_state heartbeat timestamp and optionally current_teacher."""
    _ensure_file('class_state', {"status": "SAFE", "fire_section": None, "current_teacher": None, "subject": None, 'teacher_heartbeat': None})
    state = load('class_state')
    state['teacher_heartbeat'] = datetime.datetime.now().isoformat()
    if teacher_name is not None:
        state['current_teacher'] = teacher_name
    save('class_state', state)


def reset_students_present_none(section=None):
    """Set `present` to None for all students (or only those in a section)."""
    _ensure_file('students', [])
    students = load('students')
    changed = False
    for s in students:
        if section is None or s.get('section') == section:
            if s.get('present') is not None:
                s['present'] = None
                changed = True
    if changed:
        save('students', students)


def send_absent_students_to_google_and_whatsapp(google_form_url, section=None):
    """Send absent students to Google Forms and notify parents via WhatsApp.
    
    Args:
        google_form_url: URL to Google Form (e.g., https://forms.gle/...)
        section: Optional section/class name to filter students
    
    Returns:
        dict with 'sent' and 'failed' lists
    """
    import webbrowser
    
    _ensure_file('students', [])
    students = load('students')
    
    result = {'sent': [], 'failed': []}
    
    # Find all absent students
    absent_students = []
    for s in students:
        if section is None or s.get('section') == section:
            if s.get('present') == False:  # Explicitly absent
                absent_students.append(s)
    
    if not absent_students:
        print("[INFO] No absent students found")
        return result
    
    print(f"[INFO] Found {len(absent_students)} absent student(s)")
    
    # Open Google Form for each absent student
    for student in absent_students:
        try:
            name = student.get('name', 'Unknown')
            parent_phone = student.get('parent_phone', '')
            
            # Open Google Form (teacher will fill in details)
            print(f"[GOOGLE FORM] Opening for {name}")
            webbrowser.open(google_form_url)
            
            result['sent'].append(name)
            
            # Send WhatsApp notification to parent if phone is available
            if parent_phone:
                try:
                    msg = f"Alert: Student {name} is absent today. Please contact the school."
                    send_sms(parent_phone, msg)
                    print(f"[WHATSAPP] Notified parent of {name} at {parent_phone}")
                except Exception as e:
                    print(f"[ERROR] Failed to notify parent: {e}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {student.get('name')}: {e}")
            result['failed'].append(student.get('name', 'Unknown'))
    
    return result
