import streamlit as st
from Joke_Generation import TrainedJokeGenerator


def main():
  # Initializing model
  trained_joke_generator = TrainedJokeGenerator('Model_Directory','Cleaned_Bad_Words.csv')
  # Create a text input for the user
  user_buildup = st.text_input("Enter a Buildup to your joke")

  # code for the sidebar options

  st.sidebar.text("Customize your Joke Generation :sunglasses:")

  num_jokes = st.sidebar.radio(
    'How many punchlines would you like to randomly generate?',
    ('1','2','3','4','5'), horizontal = True)

  temperature = st.sidebar.slider(
    label = 'Temperature',
    help = 'Select how random you would like your joke to be. \n'
    'Larger values correspond to more spontaneous jokes',
    min_value= .0, max_value= 5.0,
    value = 2.5
    )

  # Create a button to submit the form
  submit_button = st.button("Submit")

  # If the button was clicked
  if submit_button:
    # Get the model output
    output_list = trained_joke_generator.generate(user_buildup, num_sequences=num_jokes, temperature= temperature)

    # Display the output
    for output in output_list:
      st.write(output_list)

if __name__ == "__main__":
  main()