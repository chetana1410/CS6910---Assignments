from IPython.display import HTML as html_print
from IPython.display import display

# get html element
def cstr(s, color='black'):
	if s == ' ':
		return "<text style=color:#000;padding-left:10px;background-color:{}> </text>".format(color, s)
	else:
		return "<text style=color:#000;background-color:{}>{} </text>".format(color, s)

# get appropriate color for value
def get_clr(value):
	colors = [
          '#e1ddec', 'd6cfe6', '#ccc3de', 'bdb2d4', '#b4aacf', 
          'ae9fc9', '#a393c0', '#927fb5', '#836dab', '#765b9e', 
          '#664c95', '#5b3c8c', '59388f', '#502984', '#472476', 
          '#431d66', '#38195c', '#2f114d', '#250640', '#1e0034'
]
	value = int((value * 100) / 5)
	return colors[value]

def visualize_connectivity(sentence):
  html_format_str = '''
  <div class="container" style="
    font-family: Arial, Helvetica, sans-serif;
    border: 1px solid rgba(0, 0, 0, 0.514);
    border-radius: 8px;
    margin: 20px auto;
    padding: 20px 10px;
    background-color: rgba(250, 249, 249, 0.5);
    width: 1000px;
  ">
    <h2 style="
      text-align: center;
      margin-top: 10px;
      margin-bottom: 25px;
    ">Visualizing Attention for Predictions</h1>

    <div class="wrapper">
      <div class="left" style="
        width: 60%;
        padding: 10px;    
        margin: 0 auto;
        display: flex;
        justify-content: space-evenly;   
      ">
        <p style="margin: 8px 0">Input: <b>{}</b></p>
    '''.format(sentence)
  result, sentence1, attention_scores = evaluate(sentence)

  

  result = result.split(' ')
  predicted = ''.join(result[:result.index('<end>')])

  html_format_str += '''
        <p style="margin: 8px 0">Prediction: <b>{}</b></p>
      </div>

      <div class="right" style="
        width: 60%;
        padding: 0;    
        margin: 0 auto;      
      ">
        <table class="attention" style="
          background-color: white;
          text-align: center;
          width: 100%;
          border-collapse: collapse;
          border: 2px solid black;
        ">
          <tbody>
            <tr>
              <td style="border: 1px solid rgb(0, 0, 0, 0.75);padding: 10px 0;"><b>Character at each index</b></td>
              <td style="border: 1px solid rgb(0, 0, 0, 0.75);padding: 10px 0;"><b>Attention Visualization</b></td>
            </tr>
    '''.format(predicted)

  for i in range(len(predicted)):
    res = ''

    for j in range(len(sentence)):    
      res += cstr(sentence[j], get_clr(attention_scores[i, j + 1]))

    html_format_str += '''
            <tr>
              <td style="border: 1px solid rgb(0, 0, 0, 0.75);padding: 10px 0;"><b>Character at index {}: {}</b></td>
              <td style="border: 1px solid rgb(0, 0, 0, 0.75);padding: 10px 0;">{}</td>
            </tr>
    '''.format(i, predicted[i], res)
  
  html_format_str += '''
          </tbody>
        </table>
      </div>
    </div>
  </div>
  '''
 
  display(html_print(html_format_str))
	
visualize_connectivity('sunil')
