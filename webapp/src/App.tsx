import React, {createRef, FormEvent, useEffect, useState} from 'react';
import './App.css';
import axios from 'axios';
import dataJson from './sample/data.json'

function App() {
  const $ref = createRef<any>();
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    setData(dataJson.data);
  }, [])

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
      {data?.map((entry, index) => (
        <div key={index}>
          <h3 style={{color: 'red'}}>{index}</h3>
          <table style={{margin: "auto", marginBottom: 10, textAlign: 'left', width: 1000}}>
            <thead>
              <tr>
                <th></th>
                <th style={{width: '40%'}}>{entry?.a[0]}</th>
                <th style={{width: '55%'}}>{entry?.a[1]}</th>
              </tr>
            </thead>
            <tbody>
              {entry?.b.map((subEntry: any[], subIndex: number) => (
                <tr key={subIndex}>
                  <td>
                    <input type="checkbox" id="vehicle1" name="vehicle1" value="Bike" />
                  </td>
                  <td>{subEntry[0]}</td>
                  <td>{subEntry[1]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <br/>
        </div>
      ))}
    </div>
  );
}

export default App;
