#Creating Prices List
prices = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]
total = 0

while True:
    ask_product = int(input("Please select product (1-10) 0 to Quit:"))
    if ask_product > 0 and ask_product <= 10: #Filtering out the inputs other than product selection
        product_price = prices[ask_product - 1]
        print(f"Product: {ask_product} Price : {product_price}")
        total += product_price
    elif ask_product == 0:
        print(f"Total: {total}")
        while True:
            payment = int(input("Payment: "))
            #Creating a loop if Payment is done lower than the total price
            if payment < total:
                print(f"Insuffcient amount. Please make the payment more than {total}")
            else:
                print(f"Change: {payment - total}")
                break
        break #Exits program after successful payments and calculation
    else:
        print("Invalid Input")