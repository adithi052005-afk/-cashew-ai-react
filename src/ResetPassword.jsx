import { useState } from "react";

function ResetPassword({ onBackToLogin }) {
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [message, setMessage] = useState("");

  const token = new URLSearchParams(window.location.search).get("token");

  const handleResetPassword = async (event) => {
    event.preventDefault();

    if (password !== confirmPassword) {
      setMessage("Passwords do not match.");
      return;
    }

    try {
      const response = await fetch(
        "http://192.168.0.108:5000/reset-password",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            token: token,
            password: password,
          }),
        }
      );

      const data = await response.json();

      if (response.ok) {
        setMessage("Password reset successfully! You can now login.");
        setPassword("");
        setConfirmPassword("");
      } else {
        setMessage(data.message || "Password reset failed.");
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
          <p>Create a new password</p>
        </div>

        <form onSubmit={handleResetPassword}>

          <div className="input-group">
            <label>New Password</label>

            <input
              type="password"
              placeholder="Enter new password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
          </div>

          <div className="input-group">
            <label>Confirm Password</label>

            <input
              type="password"
              placeholder="Confirm new password"
              value={confirmPassword}
              onChange={(event) =>
                setConfirmPassword(event.target.value)
              }
              required
            />
          </div>

          <button
            type="submit"
            className="login-button"
          >
            Reset Password
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

export default ResetPassword;