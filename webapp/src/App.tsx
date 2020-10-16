import React, {createRef, FormEvent, useEffect, useState} from 'react';
import './App.css';
import axios from 'axios';

const API_ROOT = 'http://127.0.0.1:5000';

function App() {
  const $ref = createRef<any>();
  const [data, setData] = useState<any[]>([]);
  const [transactionId, setTransactionId] = useState<string>('');
  const [loadingData, setLoadingData] = useState<boolean>(false);
  const [userInput, setUserInput] = useState<string>('');

  const fetchData = () => {
    if (!transactionId) {
      return;
    }
    setLoadingData(true);
    setTimeout(() => {
      axios.post(`${API_ROOT}/status`, {
        task_id: transactionId
      }).then((response) => {
        setData(response.data.data);
        console.log('Get data successfully', response.data.data);
        setLoadingData(false);
      });
    }, 10000);
  }
  useEffect(() => {
    fetchData();
  }, [transactionId]);

  const onFileSubmit = (e: FormEvent) => {
    e.preventDefault();
    const file = $ref.current.files[0];
    const formData = new FormData();
    formData.append('file', file);
    axios.post(`${API_ROOT}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }).then((response) => {
      const taskId = response.data.task_id;
      console.log('uploaded file successfully with task id', taskId);
      setTransactionId(taskId);
    }).catch(() => {
      console.log('Upload failed');
    })
  }

  const submitData = () => {
    axios.post(`${API_ROOT}/submit`, {
      task_id: transactionId,
      user_input: userInput, // should be changeable
      data
    }).then(() => {
      console.log('Final submission is successfully');
      // if user_input is 'n' clear all data and getting back to the first step
      if (userInput === 'n') {
        setData([]);
        setTransactionId('');
      } else {
        setData([]);
        setLoadingData(true);
        setTimeout(() => {
          axios.post(`${API_ROOT}/status`, {
            task_id: transactionId
          }).then((response) => {
            setData(response.data.data);
            console.log('Get data successfully', response.data.data);
            setLoadingData(false);
          });
        }, 10000);
      }
    })
  }

  return (
    <div style={{textAlign: "center", padding: 50}}>
      <b> Welcome to Active Learning Framework for Duplicate Detection in SaaS Platform!</b>
      <hr style={{marginBottom: 50}}></hr>
      {(data && data.length === 0 && !loadingData) && (
        <form onSubmit={onFileSubmit}>
          <input type="file" name="file" ref={$ref}/>
          <button type="submit">Upload your training file (Excel)</button>
        </form>
      )}
      {loadingData && (
        <span>Loading data...</span>
      )}
      {(data && data.length > 0) && (
        <>
          {data?.map((entries, index) => (
              <div key={index}>
                <h3 style={{color: 'red'}}>{index}: Choose duplicate items if applicable.</h3>
                <table className="table">
                  <thead>
                  <tr>
                    <th></th>
                    <th style={{width: '40%'}}>{entries?.a[0]}</th>
                    <th style={{width: '55%'}}>{entries?.a[1]}</th>
                  </tr>
                  </thead>
                  <tbody>
                  {entries?.b.map((subEntry: any[], subIndex: number) => {
                    subEntry[2] = false;
                    return (
                      <tr key={subIndex}>
                        <td>
                          <input type="checkbox"
                            onChange={event => {
                              subEntry[2] = event.target.checked;
                            }}
                          />
                        </td>
                        <td>{subEntry[0]}</td>
                        <td>{subEntry[1]}</td>
                      </tr>
                    )
                  })}
                  </tbody>
                </table>
                <br/>
              </div>
            ))}
          <input type="text" placeholder="Do you want to load more? (y/N)" style={{width: 300}} value={userInput} onChange={(e) => setUserInput(e.target.value)}/>
          <button type={"button"} onClick={submitData}>Done</button>
        </>
      )}
    </div>
  );
}

export default App;
