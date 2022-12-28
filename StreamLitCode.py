import streamlit as st
from Joke_Generation import TrainedJokeGenerator


def Intialize_joke_model():
  """
  Function that caches the joke generator
  :return: Returns an instance of the TrainedJokeGenerator which will be used
  """
  return TrainedJokeGenerator('Model_Directory', 'Cleaned_Bad_Words.csv')

def main():

  st.title("AI Joke Generation using Sequence2Sequence Modelling!")

  # Initializing model and storing session state: (Only initialize model if you haven't already done this)
  if 'Model' not in st.session_state:
    st.session_state['Model'] = Intialize_joke_model()

  #Making Columns:
  col1, col2 = st.columns(2)

  # widgets for left column:
  with col1:
    st.sidebar.title("Customize your Joke Generation 😎")
    st.text('')
    num_jokes = st.sidebar.radio(
      'How many punchlines would you like to randomly generate?',
      ('1','2','3','4','5'), horizontal = True)

    temperature = st.sidebar.slider(
      label = 'Temperature',
      help = 'Select how random you would like your joke to be. \n'
      'Larger values correspond to more spontaneous jokes but v',
      min_value= 0.5, max_value= 1.5,
      value = 1.0
      )

  # Widgets for right column:
  # Put smome blank space between title and rest of page
  st.write('Disclaimer: This model was trained on reddit joke data. I do not condone or take any liability for what is generated.')
  st.write('')
  st.subheader('Type a build-up to your favourite joke!')
  user_buildup = st.text_input('Type a build-up to your favourite joke!',label_visibility='hidden')
  submit_button = st.button("Generate Punchlines!")
  st.write('')

  # If the button was clicked
  if submit_button:
    # Get the model output
    st.session_state['Generated_Punchlines'] = st.session_state['Model'].generate(user_buildup, num_sequences=int(num_jokes), temperature=temperature)

  if 'Generated_Punchlines' in st.session_state:
    st.subheader('Generated Punchlines:')
    # Display the output
    for output in st.session_state['Generated_Punchlines']:
      st.write(output)



if __name__ == "__main__":
  main()