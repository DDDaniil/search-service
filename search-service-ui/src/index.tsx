import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import {createTheme, MantineProvider} from "@mantine/core";

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

const theme = createTheme({
    /** Put your mantine theme override here */
});

root.render(
  <React.StrictMode>
      <MantineProvider theme={theme}>
        <App />
      </MantineProvider>
  </React.StrictMode>
);

