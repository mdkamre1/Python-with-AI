# Splits a sentence into parts using a seperator ' '
def my_split(sentence, separator=' '):
    result = []
    current_word = ''

    for character in sentence:
        if character == separator:
            result.append(current_word)
            current_word = ''
        else:
            current_word += character

    # Append the last word if it exists
    if current_word:
        result.append(current_word)

    return result

# Joins elements of a list (items) into a single string, using sperator ','
def my_join(items, separator=','):

    result = ''
    for i, item in enumerate(items):
        result += str(item)
        if i < len(items) - 1:  #Don't add separator after last item
            result += separator

    return result


# Main program for testing
def main():
    sentence = input("Please enter sentence:")

    # Split the sentence into words
    words = my_split(sentence)

    # Join words with commas and print
    print(my_join(words, ','))

    # Print each word on a new line
    for word in words:
        print(word)

#Run Main Function
main()