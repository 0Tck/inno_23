import { InjectModel } from '@nestjs/sequelize';
import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import { User } from './users.model';
import { CreateUserDto } from './dto/create-user.dto';

@Injectable()
export class UserService {
  constructor(
    @InjectModel(User)
    private userModel: typeof User,
  ) {}

  async createUser(payload: CreateUserDto): Promise<User> {
    try {
      let queryResponse;
      const findUser = await this.userModel.findOne({
        where: { userEmail: payload.userEmail },
      });
      if (findUser == null) {
        queryResponse = await this.userModel.create(payload);
      }
      return queryResponse;
    } catch (error) {
      console.error(error);
      throw new HttpException(
        `Error in Creating New user - ${error.name}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async userLogin(payload: {
    userEmail: string;
    password: string;
  }): Promise<any> {
    try {
      const findUser = await this.userModel.findOne({
        where: { userEmail: payload.userEmail },
      });
      if (findUser == null) {
        return { success: false, message: 'User not available' };
      } else {
        const storedpassword = findUser.password;
        if (storedpassword === payload.password) {
          //login successfull
          return { data: findUser, success: true, message: 'Login successful' };
        } else {
          //send messsage incorrect password
          return { success: false, message: 'Incorrect password' };
        }
      }
    } catch (error) {
      console.error(error);
      throw new HttpException(
        `Error in logging user - ${error.name}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async getAll(): Promise<User[]> {
    try {
      const queryResponse = await this.userModel.findAll();
      return queryResponse;
    } catch (error) {
      console.error(error);
      throw new HttpException(
        `Error in Getting All users - ${error.name}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  async getById(userId: number): Promise<User> {
    try {
      const queryResponse = await this.userModel.findOne({
        where: {
          userId: userId,
        },
      });
      return queryResponse;
    } catch (error) {
      console.error(error);
      throw new HttpException(
        `Error in Getting user with ID '${userId}' - ${error.name}`,
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }
}
