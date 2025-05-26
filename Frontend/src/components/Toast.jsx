import React from "react";

const Toast = ({ message }) => {
  return (
    <div className={`toast ${message ? "show" : ""}`}>
      {message}
    </div>
  );
};

export default Toast;
