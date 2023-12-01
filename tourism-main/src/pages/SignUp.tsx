import React, { useState } from "react";
import { createUser } from "../APISs/usersapi";
import { useNavigate } from "react-router-dom";

const SignUp: React.FC = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [displayErrorMessagePassword, setDisplayErrorMessagePassword] =
    useState(false);
  const [displayErrorMessageEmail, setDisplayErrorMessageEmail] =
    useState(false);
  const navigate = useNavigate();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    // Email validation regular expression
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));

    if (name === "confirmPassword") {
      if (value !== formData.password) {
        setDisplayErrorMessagePassword(true);
      } else {
        setDisplayErrorMessagePassword(false);
      }
    }

    if (name === "email" && !emailRegex.test(value)) {
      setDisplayErrorMessageEmail(true);
    } else {
      setDisplayErrorMessageEmail(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const payload = {
      username: formData.name,
      userEmail: formData.email,
      password: formData.password,
    };
    const data = await createUser(payload);
    navigate(`/login`);
  };

  return (
    <div className="min-h-screen flex items-center justify-center">
      <form className="max-w-md w-full p-6 bg-white rounded-md shadow-md">
        <h2 className="text-2xl font-semibold mb-6">SignUp</h2>

        <div className="mb-4">
          <label
            htmlFor="name"
            className="block text-gray-600 text-sm font-medium mb-2"
          >
            Name
          </label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            className="w-full border border-gray-300 rounded-md p-2"
          />
        </div>

        {displayErrorMessageEmail && (
          <div className="text-[12px] text-red-600">*Invalid Email</div>
        )}
        <div className="mb-4">
          <label
            htmlFor="email"
            className="block text-gray-600 text-sm font-medium mb-2"
          >
            Email
          </label>
          <input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            className="w-full border border-gray-300 rounded-md p-2"
          />
        </div>

        <div className="mb-4">
          <label
            htmlFor="password"
            className="block text-gray-600 text-sm font-medium mb-2"
          >
            New Password
          </label>
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            className="w-full border border-gray-300 rounded-md p-2"
          />
        </div>

        {displayErrorMessagePassword && (
          <div className="text-[12px] text-red-600">
            *Passwords do not match
          </div>
        )}

        <div className="mb-4">
          <label
            htmlFor="confirmPassword"
            className="block text-gray-600 text-sm font-medium mb-2"
          >
            Confirm Password
          </label>
          <input
            type="password"
            id="confirmPassword"
            name="confirmPassword"
            value={formData.confirmPassword}
            onChange={handleChange}
            className="w-full rounded-md p-2 border border-gray-300"
          />
        </div>

        <button
          type="submit"
          onClick={handleSubmit}
          className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition duration-300"
        >
          SignUp
        </button>
      </form>
    </div>
  );
};

export default SignUp;
