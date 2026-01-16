import time
from playsound import playsound

def countdown(seconds):
    while seconds > 0:
        print(f"{seconds}...", end=" ", flush=True)
        time.sleep(1)
        seconds -= 1
    print("\n")

def grounding_54321():
    print("\n🌿 5-4-3-2-1 Grounding Exercise")
    print("5 things you can see:")
    input()
    print("4 things you can touch:")
    input()
    print("3 things you can hear:")
    input()
    print("2 things you can smell:")
    input()
    print("1 thing you can taste:")
    input()
    print("\n✅ You are grounded in the present moment.")

def guided_breathing():
    print("\n🧘 Guided Breathing (4-7-8 Technique)")
    for _ in range(3):
        print("Inhale for 4 seconds")
        countdown(4)
        print("Hold for 7 seconds")
        countdown(7)
        print("Exhale for 8 seconds")
        countdown(8)
    print("\n✅ Breathing complete. Feel the calm.")

def progressive_muscle_relaxation():
    print("\n💪 Progressive Muscle Relaxation")
    print("Tense each muscle group for 5 seconds, then release.\n")
    steps = [
        "Tense your hands into fists... then release.",
        "Tense your arms... then release.",
        "Shrug your shoulders up to your ears... then release.",
        "Tense your legs... then release.",
        "Tense your feet and toes... then release.",
    ]
    for step in steps:
        print(step)
        countdown(5)
    print("\n✅ Relaxation complete. Notice the difference in your body.")

def guided_imagery():
    print("\n🌄 Guided Imagery")
    print("Close your eyes and imagine a peaceful place — a beach, forest, or mountain.\n")
    steps = [
        "Picture what you see... colors, shapes.",
        "Notice the sounds... birds, wind, waves.",
        "Feel the temperature... sun on your skin, cool breeze.",
        "Take a deep breath... and feel safe in this space.",
    ]
    for step in steps:
        print(step)
        time.sleep(5)
    print("\n✅ Visualization complete. Carry the peace with you.")

def affirmation_repetition():
    print("\n💬 Positive Affirmation Repetition")
    affirmations = [
        "I am safe in this moment.",
        "I can handle what comes my way.",
        "I deserve to feel calm.",
        "This feeling is temporary.",
        "I am grounded and in control."
    ]
    for aff in affirmations:
        print(f"🧠 {aff}")
        time.sleep(3)
    print("\n✅ Affirmations complete. Remember, your thoughts can guide your feelings.")

def counting_backwards():
    print("\n🔢 Counting Backwards from 100 by 7s")
    count = 100
    while count > 0:
        print(count)
        time.sleep(1)
        count -= 7
    print("\n✅ Great job focusing! Mental engagement helps reduce spiraling thoughts.")

def temperature_shift():
    print("\n❄️ Temperature Shift Suggestion")
    print("Try placing your hands under cold water or holding an ice cube.")
    print("This activates your parasympathetic nervous system and helps calm intense emotion.")
    time.sleep(10)
    print("\n✅ Try it next time you're overwhelmed. Cold can reset your emotional response.")

def play_audio():
    print("\n🎶 Playing soothing audio...")
    try:
        playsound("calm.wav")  # Make sure calm.wav is in the same folder as this script
    except Exception as e:
        print("⚠️ Unable to play audio:", e)

def main():
    while True:
        print("\n🎧 Grounding Techniques Timer")
        print("1. 5-4-3-2-1 Grounding Exercise")
        print("2. Guided Breathing (4-7-8)")
        print("3. Progressive Muscle Relaxation")
        print("4. Guided Imagery")
        print("5. Affirmation Repetition")
        print("6. Counting Backwards")
        print("7. Temperature Shift")
        print("8. Play soothing audio")
        print("0. Exit")
        choice = input("\nEnter your choice: ")

        if choice == "1":
            grounding_54321()
        elif choice == "2":
            guided_breathing()
        elif choice == "3":
            progressive_muscle_relaxation()
        elif choice == "4":
            guided_imagery()
        elif choice == "5":
            affirmation_repetition()
        elif choice == "6":
            counting_backwards()
        elif choice == "7":
            temperature_shift()
        elif choice == "8":
            play_audio()
        elif choice == "0":
            print("\n🌟 Stay grounded. Take care of yourself. 🌟")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
