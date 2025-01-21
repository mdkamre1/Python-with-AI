#Assigning an empty list for the items to be added
shopping_list = []

#Using while loop for infinite loop
while True:
    action = input("Would you like to \n (1) Add or \n (2) Remove items \n (3) Quit \n").strip()
    if action == "1":
        item_add = input("What will be added?")
        shopping_list.append(item_add) #Adding the item to the list
    elif action == "2":
        print(f"There are {len(shopping_list)} items in the list.")
        item_remove = int(input("Which item is deleted?"))
        shopping_list.pop(item_remove) #Removing the product to the list
    elif action == "3":
        print(f"The following items remain in the list: ")
        print("\n".join(shopping_list)) #Printing the list line by line
        break #Breaks the loop ends if user inputs 3
    else:
        print("Incorrect Selection")