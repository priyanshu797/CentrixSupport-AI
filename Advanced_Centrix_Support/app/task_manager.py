import datetime

# Function to add a task
def add_task():
    task = input("What task would you like to add? ")
    due_date = input("Enter the due date (YYYY-MM-DD HH:MM): ")
    try:
        datetime.datetime.strptime(due_date, '%Y-%m-%d %H:%M')
        with open("todo_list.txt", "a") as file:
            file.write(f"{task} | {due_date}\n")
        print("Task added successfully!")
    except ValueError:
        print("Invalid date format. Please try again.")

# Function to view all tasks
def view_tasks():
    try:
        with open("todo_list.txt", "r") as file:
            tasks = file.readlines()
            if tasks:
                print("\nYour To-Do List:")
                for i, task in enumerate(tasks, 1):
                    print(f"{i}. {task.strip()}")
            else:
                print("Your to-do list is empty!")
    except FileNotFoundError:
        print("No to-do list found. Start by adding a task.")

# Function to remove a task
def remove_task():
    view_tasks()
    try:
        task_num = int(input("Enter the number of the task you want to remove: "))
        with open("todo_list.txt", "r") as file:
            tasks = file.readlines()
        if 1 <= task_num <= len(tasks):
            removed_task = tasks.pop(task_num - 1)
            with open("todo_list.txt", "w") as file:
                file.writelines(tasks)
            print(f"Task '{removed_task.strip()}' removed successfully!")
        else:
            print("Invalid task number.")
    except (FileNotFoundError, ValueError):
        print("Error occurred. Please try again.")

# Function to check due tasks
def check_due_tasks():
    try:
        now = datetime.datetime.now()
        with open("todo_list.txt", "r") as file:
            tasks = file.readlines()
            due_tasks = [task for task in tasks if datetime.datetime.strptime(task.split('|')[1].strip(), '%Y-%m-%d %H:%M') <= now]
            if due_tasks:
                print("\nDue Tasks:")
                for task in due_tasks:
                    print(task.strip())
            else:
                print("No tasks are due!")
    except FileNotFoundError:
        print("No to-do list found. Start by adding a task.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Menu for managing tasks
def todo_menu():
    while True:
        print("\nTo-Do List Manager")
        print("1. Add a Task")
        print("2. View Tasks")
        print("3. Remove a Task")
        print("4. Check Due Tasks")
        print("5. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            add_task()
        elif choice == '2':
            view_tasks()
        elif choice == '3':
            remove_task()
        elif choice == '4':
            check_due_tasks()
        elif choice == '5':
            print("Exiting To-Do List Manager.")
            break
        else:
            print("Invalid choice. Please try again.")

# Run the menu
if __name__ == "__main__":
    todo_menu()
