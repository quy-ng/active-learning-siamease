import React, {createRef, FormEvent, useEffect, useState} from 'react';
import './App.css';
import axios from 'axios';
import dataJson from './sample/data.json'

function App() {
  const $ref = createRef<any>();
  const [data, setData] = useState<any[]>([]);
  const [transactionId, setTransactionId] = useState<number>(-1);

  useEffect(() => {
    setData(dataJson.data);
  }, []);

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
      // Do something here, set transaction id
      // setTransactionId()
      setData(response.data);
    }).catch(() => {
      console.log('Upload failed');
    })
  }
  const onDone = () => {
    //DO SOMETHING HERE
    console.log('Done');
  }

  return (
    <div style={{textAlign: "center", padding: 50}}>
      {!data && (
        <form onSubmit={onFileSubmit}>
          <input type="file" name="file" ref={$ref}/>
          <button type="submit">Submit</button>
        </form>
      )}
      {data.length > 0 && (
        <>
          {data?.map((entries, index) => (
            <div key={index}>
              <h3 style={{color: 'red'}}>{index}</h3>
              <table className="table">
                <thead>
                  <tr>
                    <th></th>
                    <th style={{width: '40%'}}>{entries?.a[0]}</th>
                    <th style={{width: '55%'}}>{entries?.a[1]}</th>
                  </tr>
                </thead>
                <tbody>
                  {entries?.b.map((subEntry: any[], subIndex: number) => (
                    <tr key={subIndex}>
                      <td>
                        <input type="checkbox"
                          onChange={event => {
                            entries[3] = event.target.checked;
                          }}
                        />
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
          <button type={"button"} onClick={onDone}>Done</button>
        </>
      )}
    </div>
  );
}

export default App;
