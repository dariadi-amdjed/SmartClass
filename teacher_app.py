import tkinter as tk
from tkinter import ttk, messagebox, font
import datetime, json, os, firebase, threading, time
import numpy as np

# Import Fire Detection System
try:
    from fire_sensor import get_fire_controller
    FIRE_SENSOR_AVAILABLE = True
except Exception:
    get_fire_controller = None
    FIRE_SENSOR_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

DATA_DIR = "data"
CAMERA_INDEX = 0

TEACHER_SCORE_THRESHOLD = 15    # minimum ORB score to consider a possible match
TEACHER_REQUIRED_SECONDS = 0.01  # seconds to maintain valid match before accepting


COLOR_PRIMARY = "#2563EB"      
COLOR_SECONDARY = "#1E40AF"   
COLOR_SUCCESS = "#059669"      
COLOR_DANGER = "#DC2626"       
COLOR_WARNING = "#D97706"      
COLOR_BG = "#F8FAFC"           
COLOR_CARD = "#FFFFFF"        
COLOR_TEXT = "#0F172A"  
COLOR_TEXT_LIGHT = "#64748B"
COLOR_ACCENT = "#0EA5E9"      

class TeacherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartClass - Teacher App")
        self.root.geometry("1400x850")
        self.root.minsize(1200, 700)
        self.root.configure(bg=COLOR_BG)

        self.teacher = None
        self.subject = None
        self.camera_on = False
        self.cap = None
        
        
        self.fire_controller = None
        self.fire_active = False
        self.fire_alarm_window = None
        
        self._setup_fonts()
        
        self._init_fire_sensor()

        self.login_screen()
    
    def _setup_fonts(self):
        """Configure custom fonts for the application"""
        self.font_title = font.Font(family="Segoe UI", size=28, weight="bold")
        self.font_heading = font.Font(family="Segoe UI", size=18, weight="bold")
        self.font_subheading = font.Font(family="Segoe UI", size=14, weight="bold")
        self.font_normal = font.Font(family="Segoe UI", size=11)
        self.font_small = font.Font(family="Segoe UI", size=10)

    def _init_fire_sensor(self):
        """Initialize the fire detection controller at app startup."""
        self.fire_controller = None
        if not FIRE_SENSOR_AVAILABLE or get_fire_controller is None:
            print("[INFO] Fire sensor module unavailable")
            return

        try:
            self.fire_controller = get_fire_controller()
            if self.fire_controller and self.fire_controller.connect():
                self.fire_controller.set_fire_detected_callback(self._on_fire_detected)
                self.fire_controller.set_fire_cleared_callback(self._on_fire_cleared)
                self.fire_controller.start_monitoring()
                print("[SUCCESS] Fire detection system initialized successfully")
            else:
                print("[ERROR] Failed to connect to Arduino")
                self.fire_controller = None
        except Exception as e:
            print(f"[ERROR] Fire system initialization failed: {e}")
            self.fire_controller = None

    # ============== LOGIN SCREEN ==============
    def login_screen(self):
        self.clear()
        
        main_frame = tk.Frame(self.root, bg=COLOR_PRIMARY)
        main_frame.pack(fill="both", expand=True)

        center_frame = tk.Frame(main_frame, bg=COLOR_PRIMARY)
        center_frame.pack(expand=True, padx=40, pady=40)

        title_label = tk.Label(center_frame, text="SmartClass", 
                              font=self.font_title, fg="white", bg=COLOR_PRIMARY)
        title_label.pack(pady=20)

        subtitle_label = tk.Label(center_frame, text="Smart Classroom Management - Teacher App", 
                                  font=("Segoe UI", 14), fg="white", bg=COLOR_PRIMARY)
        subtitle_label.pack(pady=10)

        form_frame = tk.Frame(center_frame, bg=COLOR_CARD, relief="flat", bd=0)
        form_frame.pack(pady=30, padx=0)
        
        shadow_frame = tk.Frame(main_frame, bg=COLOR_PRIMARY)
        shadow_frame.pack()

        inner_frame = tk.Frame(form_frame, bg=COLOR_CARD, padx=40, pady=30)
        inner_frame.pack()

        tk.Label(inner_frame, text="Login with Face Recognition", font=self.font_subheading, 
             fg=COLOR_TEXT, bg=COLOR_CARD).pack(anchor='center', pady=(6, 18))

        # Large face-login button
        accent = globals().get('COLOR_ACCENT', COLOR_PRIMARY)
        face_btn = tk.Button(inner_frame, text="📷 Teacher Face Login", font=self.font_subheading, 
                     bg=accent, fg="white", relief="flat", bd=0, command=self.face_login, 
                     cursor="hand2", padx=20, pady=18)
        face_btn.pack(fill='x', ipady=6, pady=6)

        # Small hint text
        tk.Label(inner_frame, text="Press the button and stand in front of the camera. If your face matches, the system will log you in automatically.", 
             font=self.font_small, fg=COLOR_TEXT_LIGHT, bg=COLOR_CARD, wraplength=480, justify='center').pack(pady=(8, 0))


        

    def login_teacher(self):
        name = getattr(self, 'teacher_name_entry', None) and self.teacher_name_entry.get().strip()
        pwd = getattr(self, 'teacher_password_entry', None) and self.teacher_password_entry.get().strip()
        if not name or not pwd:
            messagebox.showwarning('Required', 'Enter name and password')
            return
        teachers = firebase.get_teachers()
        matched = None
        for t in teachers:
            if t.get('name') == name and str(t.get('password', '')) == pwd:
                matched = t
                break
        if not matched:
            messagebox.showerror('Login Failed', 'Incorrect name or password')
            return
        self.teacher = matched.get('name')
        self.subject = matched.get('subject')
        firebase.set_current_teacher(self.teacher, self.subject)
        self.dashboard()

    # ============== FACE-LOGIN HELPERS ==============
    def _load_known_face_descriptors(self):
        """Load face images from data/teacher_faces and compute ORB descriptors."""
        faces_dir = os.path.join(os.path.dirname(__file__), 'data', 'teacher_faces')
        known = {}
        try:
            import numpy as np
        except Exception:
            np = None
        if not os.path.isdir(faces_dir):
            return known
        # prepare detector and feature extractor
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            face_cascade = None
        try:
            orb = cv2.ORB_create(500)
        except Exception:
            orb = None

        files = [f for f in os.listdir(faces_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for fn in files:
            path = os.path.join(faces_dir, fn)
            try:
                # Use imdecode instead of imread to handle Unicode paths correctly
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                roi = gray
                # try to detect face bbox and crop
                if face_cascade is not None:
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        roi = gray[y:y+h, x:x+w]
                # resize to consistent size
                try:
                    roi = cv2.resize(roi, (250, 250))
                except Exception:
                    pass
                kp, des = (None, None)
                if orb is not None:
                    kp, des = orb.detectAndCompute(roi, None)
                known[fn] = {'kp': kp, 'des': des, 'img': roi}
            except Exception:
                continue
        # build mapping filename -> teacher (from teachers.json face_image ONLY - strict mapping)
        mapping = {}
        try:
            teachers = firebase.get_teachers()
        except Exception:
            teachers = []
        
        # ONLY accept explicit 'face_image' field in teachers.json
        # No fuzzy matching or automatic pairing - must be explicit
        for t in teachers:
            fi = t.get('face_image')
            if fi and fi.strip():  # Must have a non-empty face_image value
                # compare basename only (in case teachers.json stores path or filename)
                bfi = os.path.basename(fi)
                for k in list(known.keys()):
                    if os.path.basename(k) == bfi:
                        mapping[k] = t
                        break

        return {'known': known, 'mapping': mapping}

    def _assign_image_to_teacher_dialog(self, image_filename):
        """Ask user to choose a teacher to link to image_filename, then update teachers.json accordingly."""
        try:
            teachers = firebase.get_teachers()
        except Exception:
            teachers = []

        sel = {'choice': None}

        dlg = tk.Toplevel(self.root)
        dlg.title('Link Image to Teacher')
        dlg.geometry('480x360')
        tk.Label(dlg, text=f'Choose the teacher to link this image to: {image_filename}', font=self.font_subheading).pack(pady=12, padx=12)
        list_frame = tk.Frame(dlg)
        list_frame.pack(fill='both', expand=True, padx=12, pady=6)

        var = tk.StringVar()
        for t in teachers:
            name = t.get('name') or ''
            rb = tk.Radiobutton(list_frame, text=name, variable=var, value=name, anchor='w', justify='left')
            rb.pack(fill='x', padx=6, pady=4)

        def on_ok():
            sel['choice'] = var.get()
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        btns = tk.Frame(dlg)
        btns.pack(pady=10)
        tk.Button(btns, text='Link', bg=COLOR_PRIMARY, fg='white', command=on_ok).pack(side='left', padx=8)
        tk.Button(btns, text='Cancel', command=on_cancel).pack(side='left', padx=8)

        dlg.transient(self.root)
        dlg.grab_set()
        self.root.wait_window(dlg)

        choice = sel.get('choice')
        if not choice:
            return False

        # update teachers.json: remove password and set face_image
        try:
            path = os.path.join(os.path.dirname(__file__), 'data', 'teachers.json')
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = teachers

        updated = False
        for entry in data:
            if entry.get('name') == choice:
                # remove password if present
                if 'password' in entry:
                    try:
                        del entry['password']
                    except Exception:
                        pass
                entry['face_image'] = image_filename
                updated = True
                break

        if updated:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                messagebox.showinfo('Done', f'Image {image_filename} linked to {choice} and saved in teachers.json')
                return True
            except Exception as e:
                messagebox.showwarning('Error', f'Failed to save teachers.json: {e}')
                return False
        else:
            messagebox.showwarning('Not found', 'The selected teacher was not found in the data')
            return False

    def _load_student_face_descriptors(self, section=None):
        """Load student images from data/students_img using student 'image' field in students.json.
        Apply histogram equalization to handle different lighting conditions.
        """
        faces_dir = os.path.join(os.path.dirname(__file__), 'data', 'students_img')
        known = {}  # key: student_name -> {'des':..., 'idx': index}
        try:
            students = firebase.get_students()
        except Exception as e:
            print(f"[ERROR] Failed to load students: {e}")
            students = []

        try:
            # Use more keypoints for better matching
            orb = cv2.ORB_create(1000)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # CLAHE for handling different lighting conditions
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        except Exception as e:
            print(f"[ERROR] Failed to create ORB/CLAHE or face cascade: {e}")
            return known

        for idx, s in enumerate(students):
            # Load images for students in the specified section (or all if section is '*')
            if section and s.get('section') != section:
                continue
            img_fn = s.get('image')
            if not img_fn:
                continue
            path = os.path.join(faces_dir, img_fn)
            if not os.path.isfile(path):
                print(f"[WARNING] Student image file not found: {path}")
                continue
            try:
                # Use imdecode instead of imread to handle Unicode paths correctly
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[WARNING] Failed to read image: {path}")
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Apply CLAHE to handle different lighting conditions
                gray = clahe.apply(gray)
                
                roi = gray
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    roi = gray[y:y+h, x:x+w]
                try:
                    roi = cv2.resize(roi, (250, 250))
                except Exception as e:
                    print(f"[WARNING] Failed to resize ROI for {img_fn}: {e}")
                
                # Detect keypoints and compute descriptors
                kp, des = orb.detectAndCompute(roi, None)
                if des is None:
                    print(f"[WARNING] No descriptors found for {s.get('name')} ({img_fn})")
                    continue
                
                known[s.get('name')] = {'des': des, 'idx': idx, 'phone': s.get('parent_phone')}
                print(f"[INFO] Loaded face descriptor for: {s.get('name')} (keypoints: {len(kp) if kp else 0})")
            except Exception as e:
                print(f"[ERROR] Failed to process student image {path}: {e}")
                continue
        print(f"[INFO] Total student face descriptors loaded: {len(known)}")
        return known

    def face_login(self):
        """Open camera and try to login teacher by matching live face to known faces.
        Only accept faces that match known teacher images with high confidence.
        If face is unrecognized, close camera and show error message."""
        # ensure cv2 available
        if not CV2_AVAILABLE:
            messagebox.showerror('Error', 'OpenCV is not available. Face recognition cannot run.')
            return

        data = self._load_known_face_descriptors()
        known = data.get('known', {})
        mapping = data.get('mapping', {})
        if not known:
            messagebox.showwarning('No Images', 'No teacher images found in data/teacher_faces')
            return
        
        # verify we have teacher images mapped in teachers.json
        if not mapping:
            messagebox.showwarning('Error', 'No teacher image links found. Please update teachers.json')
            return

        # prepare ORB and matcher
        try:
            orb = cv2.ORB_create(500)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        except Exception:
            messagebox.showerror('Error', 'ORB feature is not available in the installed OpenCV library')
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            messagebox.showerror('Error', 'Unable to access the camera')
            return

        messagebox.showinfo('Instructions', 'Stand in front of the camera. If your face is recognized, login will happen automatically.\nPress ESC to cancel.')

        matched_teacher = None
        faces_seen = False
        valid_match_start_time = None  # timestamp when valid match was first detected
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # resize frame for faster processing (smaller width)
                frame_small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
                gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                # looser parameters to pick up faces quickly
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in faces:
                    roi = gray[y:y+h, x:x+w]
                    try:
                        roi = cv2.resize(roi, (250, 250))
                    except Exception:
                        pass
                    kp2, des2 = orb.detectAndCompute(roi, None)
                    if des2 is None:
                        continue
                    best_fn = None
                    best_score = 0
                    for fn, obj in known.items():
                        # Only check files that are mapped to teachers
                        if fn not in mapping:
                            continue
                        des1 = obj.get('des')
                        if des1 is None:
                            continue
                        try:
                            matches = bf.knnMatch(des1, des2, k=2)
                        except Exception:
                            continue
                        good = []
                        for m_n in matches:
                            if len(m_n) < 2:
                                continue
                            m, n = m_n
                            if m.distance < 0.75 * n.distance:
                                good.append(m)
                        score = len(good)
                        if score > best_score:
                            best_score = score
                            best_fn = fn
                    # mark that we saw a face frame
                    faces_seen = True

                    # compute a distance metric by ratio of good matches to descriptor count
                    best_distance = None
                    if best_fn is not None and best_score >= TEACHER_SCORE_THRESHOLD:
                        # higher score = better match (lower distance)
                        # normalize score to 0-1 range by dividing by typical max
                        # if score >= 30, distance should be low (<0.45)
                        best_distance = 1.0 - min(1.0, best_score / 30.0)
                    # threshold for accepting match is evaluated below using globals

                    if best_score >= TEACHER_SCORE_THRESHOLD and best_fn is not None:
                        # also overlay on frame
                        try:
                            cv2.putText(frame, f"{best_fn} s:{best_score} d:{best_distance:.2f}",
                                        (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        except Exception:
                            pass

                        # check strict criteria
                        if best_distance is not None and best_distance <= 0.45:
                            teacher_obj = mapping.get(best_fn)
                            if teacher_obj:
                                # valid match found
                                current_time = time.time()
                                if valid_match_start_time is None:
                                    valid_match_start_time = current_time
                                
                                elapsed = current_time - valid_match_start_time
                                remaining = max(0, TEACHER_REQUIRED_SECONDS - elapsed)
                                
                                # draw remaining time on frame
                                try:
                                    cv2.putText(frame, f"Waiting: {remaining:.1f}s", (x, y+h+40),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                                except Exception:
                                    pass
                                
                                if elapsed >= TEACHER_REQUIRED_SECONDS:
                                    matched_teacher = teacher_obj
                                    break
                            else:
                                # mapping missing, reset
                                valid_match_start_time = None
                        else:
                            # does not satisfy distance threshold, reset
                            valid_match_start_time = None
                    else:
                        # score too low, reset
                        valid_match_start_time = None
                    # draw rectangle and show
                    color = (0, 255, 0) if best_score >= 12 else (0, 165, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    # show score for debugging and guidance
                    try:
                        cv2.putText(frame, f"score:{best_score}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        if best_score < 12:
                            cv2.putText(frame, "▫️ Try moving closer or changing the lighting", (10, frame.shape[0]-20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    except Exception:
                        pass
                cv2.imshow('Face Login - Press ESC to cancel', frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC
                    break
                if matched_teacher:
                    break
        finally:
            try:
                cap.release()
                cv2.destroyAllWindows()
            except Exception:
                pass

        if matched_teacher:
            try:
                self.teacher = matched_teacher.get('name')
                self.subject = matched_teacher.get('subject')
                firebase.set_current_teacher(self.teacher, self.subject)
                messagebox.showinfo('Done', f"Welcome {self.teacher} — Face recognition login successful")
                self.dashboard()
            except Exception as e:
                print(f"[ERROR] Teacher login post-processing failed: {e}")
                messagebox.showinfo('Done', f"Welcome {self.teacher} — Face recognition login successful")
                self.dashboard()
        else:
            if faces_seen:
                messagebox.showerror('❌ Login Failed', 'Could not recognize your face. Make sure your image is correctly registered in data/teacher_faces and linked in teachers.json')
            else:
                messagebox.showinfo('Cancelled', 'Login process was cancelled.')
        
    
    def _on_fire_detected(self):
        """Called when fire is detected"""
        print("[FIRE ALERT] Fire detected!")
        self.fire_active = True
        
        try:
            # Update the UI to show the fire alert
            if hasattr(self, 'status_label'):
                self.status_label.config(text="🔥 Notice: Fire detected!", fg="white")
                self.status_label.master.config(bg=COLOR_DANGER)
                self.status_label.config(bg=COLOR_DANGER)
            
            # Show the alert window
            self._show_fire_alarm()
        except Exception as e:
            print(f"[ERROR] Error handling fire detection: {e}")
    
    def _on_fire_cleared(self):
        """Called when fire is cleared"""
        print("[SUCCESS] Fire extinguished")
        self.fire_active = False
        
        try:
            # Update the UI
            if hasattr(self, 'status_label'):
                self.status_label.config(text="🟢 Safe - Everything is OK", fg=COLOR_TEXT)
                self.status_label.master.config(bg=COLOR_CARD)
                self.status_label.config(bg=COLOR_CARD)
            
            # Close the alert window if present
            if self.fire_alarm_window:
                try:
                    self.fire_alarm_window.destroy()
                except:
                    pass
                self.fire_alarm_window = None
        except Exception as e:
            print(f"[ERROR] Error handling fire clearance: {e}")
    
    def _show_fire_alarm(self):
        """Display the fire alarm window with bold red warning text"""
        if self.fire_alarm_window:
            return
        
        try:
            self.fire_alarm_window = tk.Toplevel(self.root)
            self.fire_alarm_window.title("🔥 Fire Alert")
            self.fire_alarm_window.geometry("800x500")
            self.fire_alarm_window.configure(bg=COLOR_DANGER)
            self.fire_alarm_window.attributes('-topmost', True)  # bring the window to the front
            
            # main title - bold alert text
            title_label = tk.Label(
                self.fire_alarm_window,
                text="🔥 Fire Detected! 🔥",
                font=("Arial", 60, "bold"),
                fg="white",
                bg=COLOR_DANGER
            )
            title_label.pack(pady=30, expand=True)
            
            # alert message
            msg_label = tk.Label(
                self.fire_alarm_window,
                text="Fire protection system active:\n✓ Alarm - active\n✓ Water pump - engaged\n✓ Windows opened",
                font=("Arial", 24, "bold"),
                fg="white",
                bg=COLOR_DANGER,
                justify="center"
            )
            msg_label.pack(pady=20, expand=True)
            
            # close button
            close_btn = tk.Button(
                self.fire_alarm_window,
                text="✓ Confirm",
                font=("Arial", 18, "bold"),
                bg="white",
                fg=COLOR_DANGER,
                relief="flat",
                bd=0,
                command=lambda: self._close_fire_alarm(),
                padx=40,
                pady=15,
                cursor="hand2"
            )
            close_btn.pack(pady=20)
            
            # bring the window to the front
            self.fire_alarm_window.lift()
            
        except Exception as e:
            print(f"[ERROR] Error displaying fire alarm window: {e}")
    
    def _close_fire_alarm(self):
        """Close the alert window"""
        if self.fire_alarm_window:
            try:
                self.fire_alarm_window.destroy()
            except:
                pass
            self.fire_alarm_window = None
    
    def toggle_fire_system(self):
        """Toggle fire detection system"""
        if not self.fire_controller:
            messagebox.showwarning("⚠️ Error", "Fire system is not connected to Arduino")
            return
        
        if self.fire_controller.is_running:
            self.fire_controller.stop_monitoring()
            self.fire_system_btn.config(text="🔘 Start Fire Detection", bg=COLOR_WARNING)
        else:
            self.fire_controller.start_monitoring()
            self.fire_system_btn.config(text="🔴 Stop Fire Detection", bg=COLOR_DANGER)

    # ============== DASHBOARD ==============
    def dashboard(self):
        self.clear()
        
        # Top Navigation Bar
        top_bar = tk.Frame(self.root, bg=COLOR_PRIMARY, height=90)
        top_bar.pack(fill="x", side="top")
        top_bar.pack_propagate(False)

        # Header content
        header_frame = tk.Frame(top_bar, bg=COLOR_PRIMARY)
        header_frame.pack(fill="both", expand=True, padx=25, pady=15)

        # Teacher info
        info_text = f"👨‍🏫 Teacher: {self.teacher} | 📚 Subject: {self.subject}"
        tk.Label(header_frame, text=info_text, font=self.font_subheading, 
                fg="white", bg=COLOR_PRIMARY).pack(anchor='w', pady=5)
        
        # Time
        time_label = tk.Label(header_frame, text=f"⏰ {datetime.datetime.now().strftime('%H:%M:%S')}", 
                             font=self.font_small, fg="white", bg=COLOR_PRIMARY)
        time_label.pack(anchor='w')
        
        # Update time every second
        self._update_time(time_label)

        # Main Content Area
        main_container = tk.Frame(self.root, bg=COLOR_BG)
        main_container.pack(fill="both", expand=True)

        # Sidebar
        sidebar = tk.Frame(main_container, bg=COLOR_CARD, width=300)
        sidebar.pack(side="right", fill="y", padx=12, pady=12)
        sidebar.pack_propagate(False)

        # Sidebar title
        tk.Label(sidebar, text="Main Menu", font=self.font_heading, 
                fg=COLOR_TEXT, bg=COLOR_CARD).pack(pady=25, padx=20)

        # Menu buttons with icons
        self._create_menu_button(sidebar, "🏠 Home", self.show_general)
        self._create_menu_button(sidebar, "👥 Students", self.show_students)
        self._create_menu_button(sidebar, "📧 Parent Message", self.write_parent_note)
        
        # Separator
        tk.Frame(sidebar, bg="#E5E7EB", height=1).pack(fill="x", pady=20, padx=20)
        
        # Logout button
        self._create_menu_button(sidebar, "🚪 Logout", self.logout, bg=COLOR_DANGER)

        # Content Area
        self.content = tk.Frame(main_container, bg=COLOR_BG)
        self.content.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        # Show initial content
        self.show_general()
        
        # Start background tasks
        self.auto_update_class_state()
        try:
            firebase.process_scheduled_notifications()
        except Exception:
            pass
        try:
            if self.teacher:
                firebase.touch_heartbeat(self.teacher)
        except Exception:
            pass
    
    def _update_time(self, label):
        """Update time label every second"""
        try:
            label.config(text=f"⏰ {datetime.datetime.now().strftime('%H:%M:%S')}")
            self.root.after(1000, lambda: self._update_time(label))
        except Exception:
            pass
    
    def _create_menu_button(self, parent, text, command, bg=COLOR_PRIMARY):
        """Create a styled menu button with hover effect"""
        btn = tk.Button(parent, text=text, font=self.font_normal, 
                       bg=bg, fg="white", relief="flat", bd=0,
                       command=command, cursor="hand2", padx=20, pady=15,
                       anchor="w", justify="left", activebackground=COLOR_SECONDARY,
                       activeforeground="white")
        btn.pack(fill="x", pady=8, padx=15)
        
        # Hover effect with smooth color change
        darker_colors = {COLOR_PRIMARY: COLOR_SECONDARY, COLOR_SUCCESS: "#047857", COLOR_DANGER: "#B91C1C"}
        darker_color = darker_colors.get(bg, COLOR_SECONDARY)
        
        def on_enter(e):
            e.widget.configure(bg=darker_color)
        
        def on_leave(e):
            e.widget.configure(bg=bg)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    # ============== GENERAL VIEW ==============
    def show_general(self):
        self.clear_content()
        
        # Header card
        header_card = tk.Frame(self.content, bg=COLOR_CARD, relief="flat", bd=0)
        header_card.pack(fill="x", padx=20, pady=(20, 30))
        
        tk.Label(header_card, text="🏠 Home", font=self.font_heading, 
                fg=COLOR_TEXT, bg=COLOR_CARD).pack(anchor='w', padx=20, pady=20)

        # Status card
        status_frame = tk.Frame(self.content, bg=COLOR_CARD, relief="flat", bd=0)
        status_frame.pack(fill="x", padx=20, pady=15)
        
        state = firebase.get_class_state()
        self.status_label = tk.Label(status_frame, 
                                    text=f"status_text", 
                                    font=self.font_subheading, fg=COLOR_TEXT, bg=COLOR_CARD)
        self.status_label.pack(anchor='w', padx=20, pady=20)

        # Hardware controls (disabled for flame-only testing)
        # The temperature/servo/pump UI is commented out to avoid accidental activation
        # while you test the flame sensor and buzzer only.
        # If you want to re-enable these controls later, remove the comments here.
        # controls_frame = tk.Frame(status_frame, bg=COLOR_CARD)
        # controls_frame.pack(anchor='e', padx=20, pady=(0,12))
        # self.temp_rotate_btn = tk.Button(controls_frame, text="°C", font=self.font_small, ...)
        # ...

        # Buttons container
        buttons_frame = tk.Frame(self.content, bg=COLOR_BG)
        buttons_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Fire System Control Card
        if FIRE_SENSOR_AVAILABLE and self.fire_controller:
            fire_card = tk.Frame(buttons_frame, bg=COLOR_CARD, relief="flat", bd=0)
            fire_card.pack(fill="x", padx=0, pady=10)
            
            fire_status = "🔴 Active" if self.fire_controller.is_running else "🔘 Inactive"
            self.fire_system_btn = tk.Button(
                fire_card,
                text=f"Fire Detection System {fire_status}\n(Toggle monitoring)",
                font=self.font_subheading,
                bg=COLOR_DANGER if self.fire_controller.is_running else COLOR_WARNING,
                fg="white",
                relief="flat",
                bd=0,
                command=self.toggle_fire_system,
                cursor="hand2",
                padx=30,
                pady=30
            )
            self.fire_system_btn.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Hover effect
            def fire_on_enter(e):
                if self.fire_controller.is_running:
                    e.widget.configure(bg="#991B1B")
                else:
                    e.widget.configure(bg="#B45309")
            
            def fire_on_leave(e):
                e.widget.configure(bg=COLOR_DANGER if self.fire_controller.is_running else COLOR_WARNING)
            
            self.fire_system_btn.bind("<Enter>", fire_on_enter)
            self.fire_system_btn.bind("<Leave>", fire_on_leave)

        # Attendance button
        att_card = tk.Frame(buttons_frame, bg=COLOR_CARD, relief="flat", bd=0)
        att_card.pack(fill="both", expand=True, padx=0, pady=10)
        
        self.att_btn = tk.Button(att_card, text="▶ Take Student Attendance\n(Camera)", 
                                font=self.font_subheading, bg=COLOR_SUCCESS, fg="white", 
                                relief="flat", bd=0, command=self.toggle_attendance, 
                                cursor="hand2", padx=30, pady=30)
        self.att_btn.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Hover effect
        self.att_btn.bind("<Enter>", lambda e: e.widget.configure(bg="#047857"))
        self.att_btn.bind("<Leave>", lambda e: e.widget.configure(bg=COLOR_SUCCESS))

        # Messages button
        msg_card = tk.Frame(buttons_frame, bg=COLOR_CARD, relief="flat", bd=0)
        msg_card.pack(fill="both", expand=True, padx=0, pady=10)
        
        msg_btn = tk.Button(msg_card, text="📧 View Logged Messages\n(Absence Messages)", 
                            font=self.font_subheading, bg=COLOR_ACCENT, fg="white", 
                            relief="flat", bd=0, command=self.show_recorded_messages, 
                            cursor="hand2", padx=30, pady=30)
        msg_btn.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Hover effect
        msg_btn.bind("<Enter>", lambda e: e.widget.configure(bg="#0284C7"))
        msg_btn.bind("<Leave>", lambda e: e.widget.configure(bg=COLOR_ACCENT))

    # ============== STUDENTS VIEW ==============
    def show_students(self):
        self.clear_content()
        
        # Header card
        header_card = tk.Frame(self.content, bg=COLOR_CARD, relief="flat", bd=0)
        header_card.pack(fill="x", padx=20, pady=(20, 10))
        
        tk.Label(header_card, text="👥 Student List", font=self.font_heading, 
                fg=COLOR_TEXT, bg=COLOR_CARD).pack(anchor='w', padx=20, pady=20)

        # Table frame
        table_frame = tk.Frame(self.content, bg=COLOR_CARD, relief="flat", bd=0)
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Table styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview', font=self.font_small, rowheight=28, 
                       background=COLOR_CARD, foreground=COLOR_TEXT, 
                       fieldbackground=COLOR_CARD, borderwidth=0)
        style.configure('Treeview.Heading', font=self.font_normal, background=COLOR_PRIMARY, 
                       foreground="white", relief="flat", borderwidth=0)
        style.map('Treeview', background=[('selected', COLOR_ACCENT)], 
                 foreground=[('selected', 'white')])
        
        table = ttk.Treeview(table_frame, columns=("first", "last", "dob", "parent", "status", "section"), 
                            show="headings", height=15)
        table.heading("first", text="First Name")
        table.heading("last", text="Last Name")
        table.heading("dob", text="DOB")
        table.heading("parent", text="Parent Phone")
        table.heading("status", text="Status")
        table.heading("section", text="Section")
        
        # Column widths
        table.column("first", width=100)
        table.column("last", width=100)
        table.column("dob", width=110)
        table.column("parent", width=140)
        table.column("status", width=90)
        table.column("section", width=110)
        
        table.pack(fill="both", expand=True, padx=10, pady=10)

        students = firebase.get_students()
        for i, s in enumerate(students):
            first = s.get('first_name') or (s.get('name') and s.get('name').split()[0]) or '-'
            last = s.get('last_name') or (s.get('name') and ' '.join(s.get('name').split()[1:])) or '-'
            status = '✅ Present' if s.get('present') is True else ('❌ Absent' if s.get('present') is False else '❓ Unknown')
            table.insert("", "end", values=(first, last, s.get("dob", "-"), s.get('parent_phone','-'), status, s.get('section')))

        self.students_table = table

        # Action buttons frame
        btn_frame = tk.Frame(self.content, bg=COLOR_CARD, relief="flat", bd=0)
        btn_frame.pack(fill="x", padx=20, pady=15)
        
        self._create_action_button(btn_frame, "➕ Add Student", self.add_student, bg=COLOR_SUCCESS)
        self._create_action_button(btn_frame, "🔄 Reset", self.restart_attendance, bg=COLOR_WARNING)
        self._create_action_button(btn_frame, "🗑️ Delete Student", self.delete_selected_student, bg=COLOR_DANGER)

        # Start refresh
        self._students_view_active = True
        self._schedule_students_refresh()
    
    def _create_action_button(self, parent, text, command, bg=COLOR_PRIMARY):
        """Create an action button with hover effect"""
        btn = tk.Button(parent, text=text, font=self.font_normal, 
                       bg=bg, fg="white", relief="flat", bd=0,
                       command=command, cursor="hand2", padx=15, pady=10,
                       activebackground=COLOR_SECONDARY, activeforeground="white")
        btn.pack(side="left", padx=5)
        
        # Hover effect colors
        hover_colors = {
            COLOR_PRIMARY: COLOR_SECONDARY, 
            COLOR_SUCCESS: "#047857", 
            COLOR_WARNING: "#B45309", 
            COLOR_DANGER: "#B91C1C"
        }
        hover_color = hover_colors.get(bg, COLOR_SECONDARY)
        
        def on_enter(e):
            e.widget.configure(bg=hover_color)
        
        def on_leave(e):
            e.widget.configure(bg=bg)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    # ============== ADD STUDENT ==============
    def add_student(self):
        win = tk.Toplevel(self.root)
        win.title("Add New Student")
        win.geometry("550x450")
        win.configure(bg=COLOR_BG)
        
        # Center the dialog over main window
        win.update_idletasks()
        try:
            rx = self.root.winfo_x()
            ry = self.root.winfo_y()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            w = 550; h = 450
            x = rx + max((rw - w)//2, 0)
            y = ry + max((rh - h)//2, 0)
            win.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass
        
        # Main frame
        main_frame = tk.Frame(win, bg=COLOR_CARD, relief="flat", bd=0)
        main_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Header
        header = tk.Frame(main_frame, bg=COLOR_PRIMARY, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        tk.Label(header, text="➕ Add New Student", font=self.font_heading, 
                fg="white", bg=COLOR_PRIMARY).pack(anchor='w', padx=25, pady=20)

        # Form content
        frm = tk.Frame(main_frame, bg=COLOR_CARD, padx=25, pady=25)
        frm.pack(fill="both", expand=True)
        
        # Name field
        tk.Label(frm, text="👤 First Name", font=self.font_subheading, fg=COLOR_TEXT, bg=COLOR_CARD).pack(anchor='w', pady=(0, 5))
        first_entry = tk.Entry(frm, font=self.font_normal)
        first_entry.pack(fill='x', ipady=10, pady=(0, 15))
        first_entry.configure(bg="white", relief="solid", bd=1, fg=COLOR_TEXT)
        
        # Last name field
        tk.Label(frm, text="👤 Last Name", font=self.font_subheading, fg=COLOR_TEXT, bg=COLOR_CARD).pack(anchor='w', pady=(0, 5))
        last_entry = tk.Entry(frm, font=self.font_normal)
        last_entry.pack(fill='x', ipady=10, pady=(0, 15))
        last_entry.configure(bg="white", relief="solid", bd=1, fg=COLOR_TEXT)
        
        # DOB field
        tk.Label(frm, text="📅 DOB (YYYY-MM-DD)", font=self.font_subheading, fg=COLOR_TEXT, bg=COLOR_CARD).pack(anchor='w', pady=(0, 5))
        dob_entry = tk.Entry(frm, font=self.font_normal)
        dob_entry.pack(fill='x', ipady=10, pady=(0, 15))
        dob_entry.configure(bg="white", relief="solid", bd=1, fg=COLOR_TEXT)
        
        # Phone field
        tk.Label(frm, text="☎️ Parent Phone", font=self.font_subheading, fg=COLOR_TEXT, bg=COLOR_CARD).pack(anchor='w', pady=(0, 5))
        parent_phone_entry = tk.Entry(frm, font=self.font_normal)
        parent_phone_entry.pack(fill='x', ipady=10, pady=(0, 25))
        parent_phone_entry.configure(bg="white", relief="solid", bd=1, fg=COLOR_TEXT)

        def save():
            students = firebase.get_students()
            section = students[0].get('section', 'Section 1') if students else 'Section 1'
            rolls = [s.get('roll', 0) for s in students if s.get('section') == section]
            next_roll = max(rolls) + 1 if rolls else 1
            first = first_entry.get().strip()
            last = last_entry.get().strip()
            
            if not first or not last:
                messagebox.showwarning("⚠️ Missing data", "Please enter first name and last name")
                return
            
            name = f"{first} {last}".strip()
            students.append({"name": name, "first_name": first, "last_name": last, "present": None, 
                           "dob": dob_entry.get(), "section": section, "roll": next_roll, 
                           'parent_phone': parent_phone_entry.get()})
            firebase.save("students", students)
            win.destroy()
            messagebox.showinfo("✅ Done", "Student added successfully")

        # Buttons
        btn_frame = tk.Frame(frm, bg=COLOR_CARD)
        btn_frame.pack(fill='x', pady=(10, 0))
        
        save_btn = tk.Button(btn_frame, text="💾 Save", font=self.font_subheading, 
                            bg=COLOR_SUCCESS, fg="white", relief="flat", bd=0,
                            command=save, cursor="hand2", padx=30, pady=10)
        save_btn.pack(side='left', fill='x', expand=True, padx=(0, 8))
        
        # Hover effect
        save_btn.bind("<Enter>", lambda e: e.widget.configure(bg="#047857"))
        save_btn.bind("<Leave>", lambda e: e.widget.configure(bg=COLOR_SUCCESS))
        
        cancel_btn = tk.Button(btn_frame, text="✕ Cancel", font=self.font_subheading, 
                              bg=COLOR_DANGER, fg="white", relief="flat", bd=0,
                              command=win.destroy, cursor="hand2", padx=30, pady=10)
        cancel_btn.pack(side='left', fill='x', expand=True)
        
        # Hover effect
        cancel_btn.bind("<Enter>", lambda e: e.widget.configure(bg="#B91C1C"))
        cancel_btn.bind("<Leave>", lambda e: e.widget.configure(bg=COLOR_DANGER))

    def delete_selected_student(self):
        sel = self.students_table.selection()
        if not sel:
            messagebox.showwarning("⚠️ No selection", "Please select a student to delete")
            return
        
        # Confirm deletion
        vals = self.students_table.item(sel[0])['values']
        name = vals[0]
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {name}?"):
            students = firebase.get_students()
            for i, s in enumerate(students):
                if s.get('name') == name or s.get('first_name') == name:
                    students.pop(i)
                    firebase.save('students', students)
                    messagebox.showinfo('✅ Done', f'{name} deleted successfully')
                    self.show_students()
                    return
            messagebox.showwarning('❌ Error', 'Student not found')
    

    # ============== PARENT NOTE ==============
    def write_parent_note(self):
        win = tk.Toplevel(self.root)
        win.title("Parent Note")
        win.geometry("650x580")
        win.configure(bg=COLOR_BG)
        win.resizable(True, True)
        
        # Main frame
        main_frame = tk.Frame(win, bg=COLOR_CARD, relief="flat", bd=0)
        main_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Header
        header = tk.Frame(main_frame, bg=COLOR_PRIMARY, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        tk.Label(header, text="📧 Parent Note", font=self.font_heading, 
                fg="white", bg=COLOR_PRIMARY).pack(anchor='w', padx=25, pady=20)

        # Form content
        frm = tk.Frame(main_frame, bg=COLOR_CARD, padx=25, pady=25)
        frm.pack(fill="both", expand=True)
        
        # Student selection
        lbl_student = tk.Label(frm, text="👤 Choose Student", font=self.font_subheading, fg=COLOR_TEXT, bg=COLOR_CARD)
        lbl_student.pack(anchor='w', pady=(0, 8))

        students = firebase.get_students()
        names = []
        for s in students:
            first = s.get('first_name') or (s.get('name') and s.get('name').split()[0]) or ''
            last = s.get('last_name') or (s.get('name') and ' '.join(s.get('name').split()[1:])) or ''
            names.append(f"{first} {last}".strip())

        selected_var = tk.StringVar()
        if names:
            selected_var.set(names[0])
        
        # Style combobox
        style = ttk.Style()
        style.configure('TCombobox', fieldbackground='white', background='white')
        
        opt = ttk.Combobox(frm, values=names, textvariable=selected_var, state='readonly', font=self.font_normal)
        opt.pack(fill='x', ipady=10, pady=(0, 20))
        opt.configure(foreground=COLOR_TEXT)

        # Note text label
        lbl_note = tk.Label(frm, text="📝 Message Content", font=self.font_subheading, fg=COLOR_TEXT, bg=COLOR_CARD)
        lbl_note.pack(anchor='w', pady=(0, 8))
        
        # Text box with border frame
        txt_frame = tk.Frame(frm, bg=COLOR_ACCENT, relief="flat", bd=0)
        txt_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        txt = tk.Text(txt_frame, height=13, font=self.font_normal, wrap="word", padx=10, pady=10)
        txt.pack(fill='both', expand=True, padx=2, pady=2)
        txt.configure(bg="white", relief="flat", bd=0, fg=COLOR_TEXT, insertbackground=COLOR_PRIMARY)

        def send_note():
            sel = selected_var.get()
            if not sel:
                messagebox.showwarning("⚠️ Selection Required", "Please choose a student")
                return
            note = txt.get("1.0", "end").strip()
            if not note:
                messagebox.showwarning("⚠️ Empty text", "Please write a message first")
                return
            
            # Find student object
            target = None
            for s in students:
                first = s.get('first_name') or (s.get('name') and s.get('name').split()[0]) or ''
                last = s.get('last_name') or (s.get('name') and ' '.join(s.get('name').split()[1:])) or ''
                if f"{first} {last}".strip() == sel:
                    target = s
                    break
            
            if not target:
                messagebox.showwarning("❌ Error", "Student not found")
                return
            
            phone = target.get('parent_phone') or target.get('phone') or target.get('parent_phone_number')
            if not phone:
                messagebox.showwarning("⚠️ No phone number", "No parent phone number available for the selected student")
                return
            
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            teacher = self.teacher or ''
            subject = self.subject or ''
            message = f"Date: {ts}\nTeacher: {teacher}\nSubject: {subject}\n---\n{note}"
            
            # Send in background thread
            try:
                threading.Thread(target=firebase.send_sms, args=(phone, message), daemon=True).start()
            except Exception:
                try:
                    firebase.send_sms(phone, message)
                except Exception:
                    pass
            
            messagebox.showinfo("✅ Sent", "Message sent successfully")
            win.destroy()

        # Buttons
        btn_frame = tk.Frame(frm, bg=COLOR_CARD)
        btn_frame.pack(fill='x', pady=(10, 0))
        
        send_btn = tk.Button(btn_frame, text="✈️ Send Message", font=self.font_subheading, 
                            bg=COLOR_PRIMARY, fg="white", relief="flat", bd=0,
                            command=send_note, cursor="hand2", padx=30, pady=12)
        send_btn.pack(side='left', padx=(0, 10), fill='x', expand=True)
        
        # Hover effect for send button
        def on_send_enter(e):
            e.widget.configure(bg=COLOR_SECONDARY)
        def on_send_leave(e):
            e.widget.configure(bg=COLOR_PRIMARY)
        
        send_btn.bind("<Enter>", on_send_enter)
        send_btn.bind("<Leave>", on_send_leave)
        
        cancel_btn = tk.Button(btn_frame, text="✕ Cancel", font=self.font_subheading, 
                              bg=COLOR_DANGER, fg="white", relief="flat", bd=0,
                              command=win.destroy, cursor="hand2", padx=30, pady=12)
        cancel_btn.pack(side='left', fill='x', expand=True)
        
        # Hover effect for cancel button
        def on_cancel_enter(e):
            e.widget.configure(bg="#B91C1C")
        def on_cancel_leave(e):
            e.widget.configure(bg=COLOR_DANGER)
        
        cancel_btn.bind("<Enter>", on_cancel_enter)
        cancel_btn.bind("<Leave>", on_cancel_leave)

    # ============== ATTENDANCE (CAMERA) ==============
    def toggle_attendance(self):
        if not CV2_AVAILABLE:
            messagebox.showerror("❌ Error", "OpenCV is not installed")
            return

        if not self.camera_on:
            self.camera_on = True
            self.att_btn.config(text="■ Stop Attendance", bg=COLOR_DANGER)
            # Start camera in background thread
            self._camera_thread = threading.Thread(target=self.start_camera, daemon=True)
            self._camera_thread.start()
            # Auto-stop after 2 minutes (120 seconds)
            self._auto_stop_timer = self.root.after(2 * 60 * 1000, self._on_auto_stop_camera)
        else:
            self.camera_on = False
            self.att_btn.config(text="▶ Take Student Attendance", bg=COLOR_SUCCESS)
            # Cancel auto-stop timer if still running
            if hasattr(self, '_auto_stop_timer'):
                try:
                    self.root.after_cancel(self._auto_stop_timer)
                except Exception:
                    pass
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _on_auto_stop_camera(self):
        """Called when 2-minute timer expires. Stops camera and sends messages."""
        if getattr(self, 'camera_on', False):
            self.camera_on = False
            try:
                if self.cap:
                    self.cap.release()
                cv2.destroyAllWindows()
            except Exception:
                pass
            try:
                self.att_btn.config(text="▶ Take Student Attendance", bg=COLOR_SUCCESS)
            except Exception:
                pass
            messagebox.showinfo('⏱️ Session ended', 'The camera closed automatically after 2 minutes. Attendance has been processed and absence messages will now be reviewed.')

    def start_camera(self):
        """Open camera for student attendance marking.
        Camera stays open until user presses button to stop or 2 minutes pass.
        On close: mark absent students and send WhatsApp messages."""

        if not CV2_AVAILABLE:
            return
        try:
            # prepare camera and face tools
            self.cap = cv2.VideoCapture(CAMERA_INDEX)  
            if not self.cap.isOpened():
                messagebox.showerror('Error', 'Unable to open the camera')
                self.camera_on = False
                return
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            # load student known descriptors
            student_known = self._load_student_face_descriptors()
            # also load teacher descriptors so we can ignore teacher faces if they appear
            teacher_data = self._load_known_face_descriptors()
            teacher_known = teacher_data.get('known', {})

            # if there are no teacher images, warn user that their own face may be
            # mistaken for a student
            if not teacher_known:
                messagebox.showwarning(
                    'Notice',
                    ('No teacher images were found in data/teacher_faces.\n'
                     'Any face appearing in front of the camera may be treated as a student if it matches a registered student image.\n'
                     'Please add your own image and link it to your name in teachers.json to avoid misidentification.')
                )

            # Check if there are any registered students with images
            if not student_known:
                faces_dir = os.path.join(os.path.dirname(__file__), "data", "students_img")
                messagebox.showwarning(
                    'Notice',
                    f'⚠️ No registered student images were found.\n\n'
                    f'Please ensure:\n'
                    f'1️⃣ Student images exist in: {faces_dir}\n'
                    f'2️⃣ student image file names are linked in students.json using the "image" field\n'
                    f'3️⃣ The file name matches the "image" field exactly\n\n'
                    f'Example:\n'
                    f'{{\n'
                    f'  "name": "Amjad Mohamed",\n'
                    f'  "image": "amjad.jpg",\n'
                    f'  "section": "Section 1"\n'
                    f'}}'
                )
                self.camera_on = False
                return
            
            try:
                orb = cv2.ORB_create(1000)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                # CLAHE for better lighting adaptation in real-time
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                print(f"[INFO] ORB (1000 keypoints), BFMatcher, and CLAHE initialized successfully")
            except Exception as e:
                print(f"[ERROR] Failed to initialize ORB/BFMatcher/CLAHE: {e}")
                orb = None
                bf = None
                clahe = None

            students = firebase.get_students()
            # Reset all students to not present at the start of attendance session
            for s in students:
                s['present'] = False
            present_set = set()  # Track students who are confirmed present in THIS session
            detected_in_frame = {}  # Track detections per frame
            student_detection_counts = {}  # Track consecutive frames where each student is detected
            REQUIRED_DETECTIONS = 1  # require detection in only 1 frame (more lenient)
            
            # Store student_known in self so we can access it after camera closes
            self._current_session_student_known = student_known

            # Main attendance loop - stays open until camera_on is False
            frame_count = 0
            while self.camera_on:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Apply CLAHE to handle different lighting conditions
                if clahe is not None:
                    gray = clahe.apply(gray)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                # prepare set of students detected in this frame
                detected_in_frame = {}
                
                # Display header with statistics
                header_text = f"Attendance Camera | Present: {len(present_set)}/{len(student_known)} | Press ESC to close"
                cv2.putText(frame, header_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press R to reset", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2)
                
                for (x, y, w, h) in faces:
                    roi = gray[y:y+h, x:x+w]
                    try:
                        roi = cv2.resize(roi, (250, 250))
                    except Exception:
                        pass
                    # compute descriptors
                    if orb is None or bf is None:
                        # no recognition available
                        continue

                    kp2, des2 = orb.detectAndCompute(roi, None)
                    if des2 is None:
                        continue

                    # first check teacher faces (ignore them)
                    if teacher_known:
                        tbest_score = 0
                        for tname, tobj in teacher_known.items():
                            tdes = tobj.get('des')
                            if tdes is None:
                                continue
                            try:
                                tmatches = bf.knnMatch(tdes, des2, k=2)
                            except Exception as e:
                                print(f"[DEBUG] Error matching teacher {tname}: {e}")
                                continue
                            tgood = []
                            for m_n in tmatches:
                                # tmatches returns lists of length k (2). skip if shorter.
                                if len(m_n) < 2:
                                    continue
                                m, n = m_n
                                if m.distance < 0.75 * n.distance:
                                    tgood.append(m)
                            tscore = len(tgood)
                            if tscore > tbest_score:
                                tbest_score = tscore
                        # ignore face if it also matches a teacher image
                        if tbest_score >= TEACHER_SCORE_THRESHOLD:
                            # draw and skip further processing
                            try:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                cv2.putText(frame, "Teacher", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                            except Exception:
                                pass
                            # reset any student detection count for this frame so teacher isn't
                            # accidentally incremented below
                            # (we don't add teacher names to detected_in_frame anywhere)
                            continue

                    # compare against known students
                    best_name = None
                    best_score = 0
                    best_distance = float('inf')
                    best_ratio = 0.0
                    for sname, obj in student_known.items():
                        des1 = obj.get('des')
                        if des1 is None:
                            continue
                        try:
                            matches = bf.knnMatch(des1, des2, k=2)
                        except Exception:
                            continue
                        good = []
                        for m_n in matches:
                            if len(m_n) < 2:
                                continue
                            m, n = m_n
                            if m.distance < 0.75 * n.distance:
                                good.append(m)
                        score = len(good)
                        if score == 0:
                            continue
                        # Compare to the smaller descriptor set so sparse live frames still qualify.
                        ratio = score / min(len(des1), len(des2)) if des2 is not None else 0.0
                        if score > best_score or (score == best_score and ratio > best_ratio):
                            if good:
                                avg_distance = sum(m.distance for m in good) / len(good)
                            else:
                                avg_distance = float('inf')
                            best_name = sname
                            best_score = score
                            best_distance = avg_distance
                            best_ratio = ratio

                    # Require sufficient matches and a reasonable distance.
                    # Student face images may produce fewer descriptors in realtime, so allow lower ratio
                    # when the absolute match count is still good.
                    MIN_MATCH_COUNT = 5
                    MIN_MATCH_RATIO = 0.14
                    MAX_AVG_DISTANCE = 90
                    if best_name is not None and best_score >= MIN_MATCH_COUNT and best_ratio >= MIN_MATCH_RATIO and best_distance < MAX_AVG_DISTANCE:
                        # Record detection in this frame
                        detected_in_frame[best_name] = True
                        print(f"[DEBUG] Detected student: {best_name} (score: {best_score}, ratio: {best_ratio:.2f}, avg_distance: {best_distance:.2f})")
                        
                        # Increment detection count for this student
                        if best_name not in student_detection_counts:
                            student_detection_counts[best_name] = 0
                        student_detection_counts[best_name] += 1
                        detected_in_frame[best_name] = True
                        
                        # find student index from name
                        try:
                            idx = None
                            for i, s in enumerate(students):
                                if s.get('name') == best_name:
                                    idx = i
                                    break
                            if idx is not None:
                                # Mark present immediately on first good detection (REQUIRED_DETECTIONS = 1)
                                if student_detection_counts.get(best_name, 0) >= REQUIRED_DETECTIONS:
                                    # if already marked, show repeated notice
                                    if students[idx].get('present') is True:
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                        cv2.putText(frame, "✅ " + best_name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                                        cv2.putText(frame, "(Already registered attendance)", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                                    else:
                                        students[idx]['present'] = True
                                        firebase.cancel_scheduled_for_student(students[idx].get('name'))
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                        cv2.putText(frame, "‏👤 " + best_name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                                        cv2.putText(frame, "✅ Welcome!", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    present_set.add(idx)
                                    continue
                        except Exception:
                            pass
                    # no confident match — draw orange rectangle
                    # Show score for debugging
                    score_text = f"❓ {best_name if best_name else 'Unknown'} ({best_score})"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(frame, score_text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # after processing all faces in this frame, remove any counts for names not seen
                for name in list(student_detection_counts.keys()):
                    if name not in detected_in_frame:
                        del student_detection_counts[name]
                
                try:
                    cv2.putText(frame, 'Press ESC to close camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Attendance - Scan Students", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        break
                    elif key == ord('r'):
                        # reset all marks in this session
                        for s in students:
                            s['present'] = False
                        present_set.clear()
                        student_detection_counts.clear()
                        print("[ATTENDANCE] Session reset by user (r key)")
                        # show short feedback on frame
                        cv2.putText(frame, 'Resetting...', (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                except Exception:
                    pass
        except Exception as e:
            print(f"[ERROR] Camera error: {e}")
        finally:
            try:
                if self.cap is not None:
                    self.cap.release()
                cv2.destroyAllWindows()
            except Exception:
                pass

        # camera closed — save attendance and send messages
        try:
            # If any students were detected even once but didn't reach the
            # required consecutive-frame threshold, still consider them present
            # once the camera is stopped. This helps when the user quickly
            # shows their face and then manually closes the camera.
            for name, cnt in student_detection_counts.items():
                if cnt > 0:
                    for s in students:
                        if s.get('name') == name:
                            s['present'] = True
            
            # Save the final attendance state (the one we updated during camera session)
            try:
                firebase.save("students", students)
                print("[ATTENDANCE] Final attendance state saved")
            except Exception as e:
                print(f"[ERROR] Failed to save final attendance: {e}")
            
            section = students[0].get('section', 'Section 1') if students else 'Section 1'
            to_send = []  # list of (phone, message) tuples
            
            # Get list of students with registered images
            registered_students = set(self._current_session_student_known.keys()) if hasattr(self, '_current_session_student_known') else set()
            
            # Count present and absent
            present_count = 0
            absent_count = 0
            
            # Use the SAME students list we just saved (don't reload from Firebase)
            for s in students:
                if s.get('section') == section:
                    # Mark student as absent if they don't have a registered image
                    # (they could not be detected anyway without an image)
                    if s.get('name') not in registered_students:
                        s['present'] = False
                    
                    # Count present/absent
                    if s.get('present') is True:
                        present_count += 1
                    else:
                        absent_count += 1
                    
                    # Send message if student is marked as absent AND has a phone number
                    if s.get('present') is not True:
                        phone = s.get('parent_phone')
                        name = s.get('name')
                        # Only send if both phone and name exist and phone is not empty
                        if phone and phone.strip() and name and name.strip():
                            msg = f"Student {name} is absent today in section {section}."
                            to_send.append((phone, msg))
            
            # save updated attendance
            try:
                firebase.save('students', students)
            except Exception as e:
                print(f"[ERROR] Error saving students: {e}")
            
            # Show attendance summary
            summary_msg = f"""
📊 Attendance Summary
{'='*30}
✅ Present: {present_count}
❌ Absent: {absent_count}
📝 Total: {present_count + absent_count}
{'='*30}
            """.strip()
            messagebox.showinfo('Attendance Summary', summary_msg)
            print(f"[ATTENDANCE SUMMARY] Present: {present_count}, Absent: {absent_count}")
            
            # Send messages directly without review dialog
            if to_send:
                print(f"[INFO] Sending {len(to_send)} absence message(s) via WhatsApp...")
                sent_count = 0
                failed_count = 0
                for phone, msg in to_send:
                    try:
                        print(f"[INFO] Sending message to {phone}: {msg}")
                        # open WhatsApp to send message
                        threading.Thread(target=firebase.send_sms, args=(phone, msg), daemon=True).start()
                        sent_count += 1
                    except Exception as e:
                        print(f"[ERROR] Error starting thread for {phone}: {e}")
                        try:
                            firebase.send_sms(phone, msg)
                            sent_count += 1
                        except Exception as e:
                            print(f"[ERROR] Error sending message to {phone}: {e}")
                            failed_count += 1
                # Show summary
                result_msg = f"""
✅ Messages Sent!
{'='*30}
📤 Success: {sent_count}
❌ Failed: {failed_count}
                """.strip()
                messagebox.showinfo('Send Result', result_msg)
            else:
                messagebox.showinfo('Send Result', 'No phone numbers available to send absence messages.')
        except Exception as e:
            print(f"[ERROR] Message sending error: {e}")

    def auto_update_class_state(self):
        # Update class state heartbeat and monitor notifications
        try:
            if self.teacher:
                firebase.touch_heartbeat(self.teacher)
        except Exception:
            pass
        # process scheduled notifications (absent -> send after delay)
        try:
            firebase.process_scheduled_notifications()
        except Exception:
            pass
        self.root.after(30000, self.auto_update_class_state)


    def _schedule_students_refresh(self):
        try:
            if getattr(self, '_students_view_active', False):
                self._refresh_students_table()
                self.root.after(5000, self._schedule_students_refresh)
        except Exception:
            pass

    def _refresh_students_table(self):
        try:
            # update the table contents
            table = getattr(self, 'students_table', None)
            if not table:
                return
            for i in table.get_children():
                table.delete(i)
            students = firebase.get_students()
            for s in students:
                first = s.get('first_name') or (s.get('name') and s.get('name').split()[0]) or '-'
                last = s.get('last_name') or (s.get('name') and ' '.join(s.get('name').split()[1:])) or '-'
                status = 'Present' if s.get('present') is True else ('Absent' if s.get('present') is False else 'Not set')
                table.insert("", "end", values=(first, last, s.get("dob", "-"), s.get('parent_phone','-'), status, s.get('section')))
        except Exception:
            pass

    # ============== FIRE EFFECTS ==============
    def start_fire_effect(self):
        """Create translucent overlay and start flashing for fire alert"""
        if getattr(self, 'fire_overlay', None) and self.fire_overlay.winfo_exists():
            return
        try:
            ov = tk.Toplevel(self.root)
            ov.overrideredirect(True)
            ov.attributes('-topmost', True)
            ov.attributes('-alpha', 0.35)
            # Place over main window
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            ov.geometry(f"{rw}x{rh}+{rx}+{ry}")
            ov.configure(bg='#ff0000')
            self.fire_overlay = ov
            self._fire_flash_on = True
            self._fire_flash_job = None
            self._flash_overlay()
        except Exception:
            pass

    def _flash_overlay(self):
        # toggle overlay visible/hidden to create flashing effect
        try:
            if not getattr(self, 'fire_overlay', None):
                return
            if self._fire_flash_on:
                self.fire_overlay.deiconify()
            else:
                self.fire_overlay.withdraw()
            self._fire_flash_on = not self._fire_flash_on
            self._fire_flash_job = self.root.after(600, self._flash_overlay)
        except Exception:
            pass

    def stop_fire_effect(self):
        try:
            if getattr(self, '_fire_flash_job', None):
                self.root.after_cancel(self._fire_flash_job)
                self._fire_flash_job = None
            if getattr(self, 'fire_overlay', None) and self.fire_overlay.winfo_exists():
                self.fire_overlay.destroy()
            self.fire_overlay = None
        except Exception:
            pass

    # ============== Hardware control helpers ==============

    # ============== LOGOUT ==============
    def logout(self):
        """Logout — do NOT send absent notifications here (only when closing attendance camera)"""
        try:
            # just logout without marking absent or sending messages
            # (messages only sent when closing camera after attendance session)
            pass
        except Exception:
            pass
        firebase.logout_teacher()
        self.login_screen()

    # ============== UTILITIES ==============
    def clear(self):
        """Clear all widgets from root"""
        for w in self.root.winfo_children():
            w.destroy()
    
    def clear_content(self):
        """Clear content area and stop student refresh"""
        try:
            self._students_view_active = False
        except Exception:
            pass
        for w in self.content.winfo_children():
            w.destroy()


    def process_absentees_after_camera(self):
        """Send absent notifications after camera session"""
        try:
            students = firebase.get_students()
            to_send = []
            for s in students:
                if s.get('present') is not True:
                    s['present'] = False
                    phone = s.get('parent_phone')
                    if phone:
                        section = s.get('section', 'Unknown')
                        msg = f"🔔 Absence notice: Student {s.get('name')} did not attend today in section {section}"
                        to_send.append((phone, msg))
            firebase.save('students', students)
            # Send messages in background
            for phone, msg in to_send:
                try:
                    threading.Thread(target=firebase.send_sms, args=(phone, msg), daemon=True).start()
                except Exception:
                    try:
                        firebase.send_sms(phone, msg)
                    except Exception:
                        pass
        except Exception:
            pass


    def restart_attendance(self):
        """Reset attendance status for all students"""
        try:
            students = firebase.get_students()
            for s in students:
                s['present'] = None
            firebase.save('students', students)
            messagebox.showinfo('✅ Done', 'Attendance status reset for all students')
            self.show_students()
        except Exception:
            messagebox.showerror('❌ Error', 'Failed to reset')
    
    def show_recorded_messages(self):
        """Show recorded absence messages for manual sending"""
        try:
            messages = firebase.load('sent_messages')
        except Exception:
            messages = []
        
        if not messages:
            messagebox.showinfo('No messages', 'No absence messages have been logged yet.')
            return
        
        # Create window
        win = tk.Toplevel(self.root)
        win.title("View Logged Messages")
        win.geometry("800x600")
        win.configure(bg=COLOR_BG)
        
        # Main frame
        main_frame = tk.Frame(win, bg=COLOR_CARD, relief="flat", bd=0)
        main_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Header
        header = tk.Frame(main_frame, bg=COLOR_PRIMARY, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        tk.Label(header, text="📧 Logged Messages", font=self.font_heading, 
                fg="white", bg=COLOR_PRIMARY).pack(anchor='w', padx=25, pady=20)
        
        # Content frame with scrollbar
        content_frame = tk.Frame(main_frame, bg=COLOR_CARD)
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create text widget with scrollbar
        scrollbar = tk.Scrollbar(content_frame)
        scrollbar.pack(side="right", fill="y")
        
        text_widget = tk.Text(content_frame, yscrollcommand=scrollbar.set, 
                             font=self.font_small, wrap="word", padx=10, pady=10)
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Add messages to text widget
        for i, msg in enumerate(messages, 1):
            timestamp = msg.get('timestamp', 'Unknown')
            phone = msg.get('to', 'Unknown')
            text = msg.get('message', '')
            status = msg.get('status', 'pending')
            
            # Format message
            formatted = f"[{i}] Phone: {phone}\nTime: {timestamp}\nStatus: {status}\nMessage:\n{text}\n" + "-"*60 + "\n\n"
            text_widget.insert(tk.END, formatted)
        
        text_widget.config(state="disabled")  # Read-only
        
        # Buttons frame
        btn_frame = tk.Frame(main_frame, bg=COLOR_CARD)
        btn_frame.pack(fill="x", padx=20, pady=15)
        
        # Copy to clipboard button
        def copy_to_clipboard():
            import subprocess
            try:
                text_content = text_widget.get("1.0", tk.END)
                # Use Windows clipboard
                process = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
                process.communicate(text_content.encode('utf-8'))
                messagebox.showinfo('✅ Done', 'Messages copied to clipboard')
            except Exception as e:
                messagebox.showerror('Error', f'Failed to copy messages: {e}')
        
        copy_btn = tk.Button(btn_frame, text="📋 Copy All", font=self.font_normal, 
                            bg=COLOR_PRIMARY, fg="white", relief="flat", bd=0,
                            command=copy_to_clipboard, cursor="hand2", padx=20, pady=10)
        copy_btn.pack(side="left", padx=5)
        
        copy_btn.bind("<Enter>", lambda e: e.widget.configure(bg=COLOR_SECONDARY))
        copy_btn.bind("<Leave>", lambda e: e.widget.configure(bg=COLOR_PRIMARY))
        
        # Close button
        close_btn = tk.Button(btn_frame, text="✕ Close", font=self.font_normal, 
                             bg=COLOR_DANGER, fg="white", relief="flat", bd=0,
                             command=win.destroy, cursor="hand2", padx=20, pady=10)
        close_btn.pack(side="left", padx=5)
        
        close_btn.bind("<Enter>", lambda e: e.widget.configure(bg="#B91C1C"))
        close_btn.bind("<Leave>", lambda e: e.widget.configure(bg=COLOR_DANGER))
    
    def cleanup(self):
        """Clean up resources when closing the app"""
        try:
            if self.fire_controller:
                self.fire_controller.stop_monitoring()
                self.fire_controller.disconnect()
                print("[SUCCESS] Fire system stopped successfully")
        except Exception as e:
            print(f"[ERROR] Error cleaning up fire system: {e}")
        
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
        except:
            pass

# ============== ENTRY POINT ==============
if __name__ == "__main__":
    root = tk.Tk()
    app = TeacherApp(root)
    
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
