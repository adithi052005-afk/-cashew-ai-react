import { useState } from "react";

function Register({ onBackToLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [message, setMessage] = useState("");

  const handleRegister = async (event) => {
    event.preventDefault();

    if (password !== confirmPassword) {
      setMessage("Passwords do not match.");
      return;
    }

    try {
      const response = await fetch("http://192.168.0.108:5000/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: email.trim(),
          password: password,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessage("Account created successfully! You can now login.");
        setEmail("");
        setPassword("");
        setConfirmPassword("");
      } else {
        setMessage(data.message || "Registration failed.");
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
          <p>Create your account</p>
        </div>

        <form onSubmit={handleRegister}>

          <div className="input-group">
            <label>Email</label>
            <input
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
          </div>

          <div className="input-group">
            <label>Password</label>
            <input
              type="password"
              placeholder="Create a password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />
          </div>

          <div className="input-group">
            <label>Confirm Password</label>
            <input
              type="password"
              placeholder="Confirm your password"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              required
            />
          </div>

          <button type="submit" className="login-button">
            Create Account
          </button>

          {message && (
            <p className="login-message">
              {message}
            </p>
          )}

        </form>

        <p className="signup-text">
          Already have an account?{" "}
          <button
            type="button"
            className="signup-button"
            onClick={onBackToLogin}
          >
            Login
          </button>
        </p>

      </div>
    </div>
  );
}

export default Register;