import moon
import planet
import stack

def show_menu():
    print("Welcome to Astro Image Processor!")
    print()
    print("1. Planetary Enhancement")
    print("2. Moon Enhancement")
    print("3. Stack Images")
    print("0. Exit")

def main():
    while True:
        show_menu()
        choice = input("Choose a mode number: ")

        if choice == "1":
            image_path = input("Enter the image path: ").strip()
            planet.process(image_path)

        elif choice == "2":
            image_path = input("Enter the image path: ").strip()
            moon.process(image_path)

        elif choice == "3":
            stack.process()

        elif choice == "0":
            print("Exiting.")
            break

        else:
            print("Invalid choice. Please try again.")


main()
