def generate_self_care_plan():
    print("\nLet's create your personalized self-care plan! Please answer the following questions.\n")

    # Collect user input with validation
    def get_input(prompt, valid_options):
        while True:
            response = input(prompt).strip().lower()
            if response in valid_options:
                return response
            print(f"Please enter one of the following options: {', '.join(valid_options)}")

    sleep_quality = get_input("How would you rate your sleep quality? (poor/average/good): ", ["poor", "average", "good"])
    diet = get_input("How healthy is your diet? (poor/average/good): ", ["poor", "average", "good"])
    physical_activity = get_input("How often do you exercise? (none/1-2 times per week/3+ times per week): ", ["none", "1-2 times per week", "3+ times per week"])
    social_connection = get_input("How connected do you feel with friends or family? (not connected/somewhat connected/very connected): ", ["not connected", "somewhat connected", "very connected"])

    plan = "\n🌟 Your Personalized Self-Care Plan 🌟\n"

# Sleep hygiene tips
    if sleep_quality == "poor":
        plan += (
            "- Sleep Hygiene: Sleep is the foundation of your well-being. Try to set a relaxing bedtime routine—dim lights, "
            "no screens at least an hour before sleep, and maybe a warm cup of herbal tea. Aim to go to bed and wake up at "
            "the same time daily, even on weekends. Little changes can create restful nights and energized days.\n\n"
        )
    elif sleep_quality == "average":
        plan += (
            "- Sleep Hygiene: You're doing okay with your sleep, which is great! To enhance it further, consider incorporating "
            "gentle relaxation techniques before bedtime, like deep breathing or light stretching. Your body and mind deserve "
            "this calm transition.\n\n"
        )
    else:
        plan += (
            "- Sleep Hygiene: Fantastic! Your good sleep habits are your superpower. Keep protecting this sacred rest time—it "
            "fuels your mind, mood, and body for all the amazing things you do.\n\n"
        )

    # Nutrition tips
    if diet == "poor":
        plan += (
            "- Nutrition: Food is your fuel and medicine. Start by adding colorful fruits and vegetables to your meals — they "
            "bring vitamins and energy that your body loves. Limit processed and sugary snacks and hydrate well. Every healthy "
            "bite is a step toward a stronger, happier you.\n\n"
        )
    elif diet == "average":
        plan += (
            "- Nutrition: You're on the right track! Try mixing up your meals with new healthy options like nuts, seeds, and whole "
            "grains. Small tweaks like choosing water over sugary drinks can make a big impact. Nourish your body, and it will "
            "thank you.\n\n"
        )
    else:
        plan += (
            "- Nutrition: Excellent work maintaining a balanced diet! Your mindful choices support your mental and physical health, "
            "helping you feel vibrant and focused every day. Keep it up and enjoy the benefits.\n\n"
        )

    # Physical activity tips
    if physical_activity == "none":
        plan += (
            "- Physical Activity: Movement is medicine for your mind. Begin gently—try a 10-minute walk in fresh air or simple "
            "stretching. Celebrate these first steps; consistency builds strength and resilience. Soon, you’ll feel more energized "
            "and confident.\n\n"
        )
    elif physical_activity == "1-2 times per week":
        plan += (
            "- Physical Activity: Great job getting active! To boost mood and reduce stress even more, aim to increase activity "
            "to at least 3 times per week. Try mixing in different activities you enjoy to keep it fun and rewarding.\n\n"
        )
    else:
        plan += (
            "- Physical Activity: Awesome dedication! Regular exercise releases those wonderful endorphins that brighten your day "
            "and build mental toughness. Keep challenging yourself and enjoy the journey of strength and well-being.\n\n"
        )

    # Social connection tips
    if social_connection == "not connected":
        plan += (
            "- Social Connection: Connection is healing. Reach out today — even a short message or smile can spark warmth. Consider "
            "joining a group or activity that interests you; new friendships can blossom where you least expect. You are not alone, "
            "and meaningful bonds nurture your soul.\n\n"
        )
    elif social_connection == "somewhat connected":
        plan += (
            "- Social Connection: You’re nurturing your relationships, which is wonderful. Plan a catch-up call or a coffee meet-up "
            "soon to deepen those ties. Sharing your thoughts and laughter strengthens your support system and uplifts your spirit.\n\n"
        )
    else:
        plan += (
            "- Social Connection: Wonderful! Your strong social ties are a true source of resilience and joy. Keep investing time and "
            "love into these bonds—they enrich your life and help you flourish.\n\n"
        )

    plan += "🌈 Remember, self-care is a journey, not a destination. Small, kind steps every day create big, lasting changes. You’re doing great! 🌈\n"

    print(plan)



def main():
    print("Welcome to your Mental Health Chatbot!\n")
    while True:
        print("Choose an option:")
        print("1. Get Personalized Self-Care Plan")
        print("0. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            generate_self_care_plan()
        elif choice == "0":
            print("\nThank you for using the Mental Health Chatbot. Take care!")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
