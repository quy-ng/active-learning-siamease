import React, {createRef, FormEvent, useState} from 'react';
import './App.css';
import axios from 'axios';
import dataJson from './sample/data.json'

function App() {
  const $ref = createRef<any>();
  console.log(dataJson);
  const [data, setData] = useState<any[][] | null>([[1, 2, 3]]);
  const [currentIndex, setCurrentIndex] = useState<number>(0);

  const onFileSubmit = (e: FormEvent) => {
    e.preventDefault();
    const file = $ref.current.files[0];
    const formData = new FormData();
    formData.append('file', file);
    axios.post('/upload-csv', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }).then((response) => {
      console.log('Upload successfully');
      setData(response.data);
    }).catch(() => {
      console.log('Upload failed');
    })
  }

  return (
    <div style={{textAlign: "center", padding: 50}}>
      {!data && (
        <form onSubmit={onFileSubmit}>
          <input type="file" name="file" ref={$ref}/>
          <button type="submit">Submit</button>
        </form>
      )}
      {(data && data.length > 0) && (
        <div>
          <table style={{margin: "auto", marginBottom: 10}}>
            <tr>
              <th>Firstname</th>
              <th>Lastname</th>
              <th>Age</th>
            </tr>
            <tr>
              <td>Jill</td>
              <td>Smith</td>
              <td>50</td>
            </tr>
            <tr>
              <td>Eve</td>
              <td>Jackson</td>
              <td>94</td>
            </tr>
          </table>
          <button type={"button"}>Next</button>
        </div>
      )}
    </div>
  );
}

export default App;
