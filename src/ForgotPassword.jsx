import { useState } from "react";

function ForgotPassword({ onBackToLogin }) {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");

  const handleForgotPassword = async (event) => {
    event.preventDefault();

    setMessage("Checking email...");

    try {
      const response = await fetch(
        "http://192.168.0.107:5000/forgot-password",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            email: email.trim(),
          }),
        }
      );

      const data = await response.json();

      if (response.ok) {
        setMessage(data.message);
      } else {
        setMessage(data.message || "Something went wrong.");
      }
    } catch (error) {
      console.error(error);
      setMessage("Cannot connect to server.");
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">

        <div className="login-header">
          <h1>Cashew AI</h1>
          <p>Reset your password</p>
        </div>

        <form onSubmit={handleForgotPassword}>

          <div className="input-group">
            <label>Email</label>

            <input
              type="email"
              placeholder="Enter your registered email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
          </div>

          <button
            type="submit"
            className="login-button"
          >
            Send Reset Link
          </button>

          {message && (
            <p className="login-message">
              {message}
            </p>
          )}

        </form>

        <p className="signup-text">
          Remember your password?{" "}

          <button
            type="button"
            className="signup-button"
            onClick={onBackToLogin}
          >
            Back to Login
          </button>

        </p>

      </div>
    </div>
  );
}

export default ForgotPassword;