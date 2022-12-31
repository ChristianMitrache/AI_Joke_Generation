import streamlit as st
from Joke_Generation import TrainedJokeGenerator


def Intialize_joke_model():
  """
  Function that caches the joke generator
  :return: Returns an instance of the TrainedJokeGenerator which will be used
  """
  return TrainedJokeGenerator('Model_Directory/Checkpoint-2300_Dropout_0.2', 'Cleaned_Bad_Words.csv')

def main():

  st.title("AI Joke Generation using Sequence2Sequence Modelling!")
  st.write('By Christian Mitrache')

  # Initializing model and storing session state: (Only initialize model if you haven't already done this)
  if 'Model' not in st.session_state:
    with st.spinner(text="Loading Language Model.... This may take a minute."):
      st.session_state['Model'] = Intialize_joke_model()

  # widgets for left column:
  st.sidebar.title("Customize your Joke Generation 😎")
  st.sidebar.markdown("---")

  # Logic for number of punchlines:
  st.sidebar.markdown('### How many punchlines would you like to generate?')
  num_jokes = st.sidebar.radio(
    'How many punchlines would you like to randomly generate?',
    ('1','2','3','4','5'), horizontal = True, label_visibility= 'hidden')
  st.sidebar.markdown("---")

  # Logic for Temperature:
  st.sidebar.markdown('### Temperature')
  st.sidebar.write('Select how random you would like the responses to be. \n'
    'Larger values correspond to more variety in punchlines but also increase the chance of incoherent generations.')
  temperature = st.sidebar.slider(
    label = 'Temperature',
    min_value= 0.25, max_value= 1.5,
    value = 1.0,
    label_visibility= 'hidden'
    )
  st.sidebar.markdown("---")

  # Logic for CensorShip:
  st.sidebar.markdown('### Censorship')
  st.sidebar.write('Censor your jokes by adding bad words or phrases that you wish the joke generator to avoid.')
  st.sidebar.write('Example: Big Fat Meany')
  censor_word_input = st.sidebar.text_input(label = 'Topic',label_visibility= 'hidden')
  censor_button = st.sidebar.button('Add to Censor List')
  if censor_button:
    # Updating bad word list:
    st.session_state['Model'].add_extra_bad_words(censor_word_input.lstrip().rstrip())
    st.sidebar.success('Added!')

  # Logic for Top K parameter:
  st.sidebar.markdown("---")
  st.sidebar.markdown('### Top-K')
  st.sidebar.write('When sampling the next work in the text generation, only considers the top K most likely words.')
  top_k = st.sidebar.radio(index = 2,
    label = 'When sampling the next work in the text generation, only considers the top K most likely words.',
    options =('3', '5', '10', '25', '50', '100'), horizontal=True, label_visibility='hidden')

  # Widgets for right column:
  # Put smome blank space between title and rest of page
  st.markdown('**Disclaimer**: &nbsp; This model was trained on reddit joke data. I do not condone or take any liability for what is generated.')
  st.write('')
  st.subheader('Type a build-up to your favourite joke!')
  user_buildup = st.text_input('Type a build-up to your favourite joke!',label_visibility='hidden')
  submit_button = st.button("Generate Punchlines!")
  st.write('')

  # If the button was clicked
  if submit_button:
    # Get the model output
    with st.spinner(text="Generating Punchlines.... \n"):
      st.session_state['Generated_Punchlines'] = st.session_state['Model'].generate(user_buildup,
                                                                                    num_sequences=int(num_jokes),
                                                                                    temperature=temperature,
                                                                                    top_k = int(top_k))

  if 'Generated_Punchlines' in st.session_state:
    st.subheader('Generated Punchlines:')
    # Display the output
    for output in st.session_state['Generated_Punchlines']:
      st.write(output)



if __name__ == "__main__":
  main()