import axios from "axios";

const API_URL = "http://localhost:3002";
export const createUser = async (userDto: any) => {
  try {
    const response = await axios.post(`${API_URL}/user/create`, userDto);
    return response.data;
  } catch (error) {
    console.error("Error creating user:", error);
    throw error;
  }
};

export const userLogin = async (logindto: any) => {
  try {
    const response = await axios.post(`${API_URL}/user/login`, logindto);
    return response.data;
  } catch (error) {
    console.error("Error creating user:", error);
    throw error;
  }
};

export const getAllUsers = async () => {
  try {
    const response = await axios.post(`${API_URL}/user/get`);
    return response.data;
  } catch (error) {
    console.error("Error in getting all users:", error);
    throw error;
  }
};

export const getUserById = async (userId: any) => {
  try {
    const response = await axios.post(`${API_URL}/user/get/${userId}`);
    return response.data;
  } catch (error) {
    console.error(`Error in getting user: with ID ${userId}`, error);
    throw error;
  }
};
