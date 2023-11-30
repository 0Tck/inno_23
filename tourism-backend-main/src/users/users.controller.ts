import {
  Controller,
  UseInterceptors,
  Body,
  Param,
  Post,
  Get,
  Put,
  Delete,
  HttpStatus,
  HttpException,
} from '@nestjs/common';
import { ApiTags } from '@nestjs/swagger';

import { FormatResponseInterceptor } from 'src/common/format-response.interceptor';
import { UserService } from './users.service';
import { CreateUserDto, LoginUserDto } from './dto/create-user.dto';
import { User } from './users.model';

@Controller('user')
@UseInterceptors(FormatResponseInterceptor)
@ApiTags('User')
export class UserController {
  constructor(private userService: UserService) {}

  @Post('create')
  async createUser(@Body() payload: CreateUserDto): Promise<User> {
    return this.userService.createUser(payload);
  }

  @Post('login')
  async loginUser(@Body() payload: LoginUserDto): Promise<any> {
    return this.userService.userLogin(payload);
  }

  @Get('get')
  async getAll(): Promise<User[]> {
    const records = await this.userService.getAll();
    if (records.length === 0) {
      throw new HttpException(
        'No user Records Found to Fetch',
        HttpStatus.NOT_FOUND,
      );
    }
    return records;
  }

  @Get('get/:userId')
  async getById(@Param('userId') userId: number): Promise<User> {
    const record = await this.userService.getById(userId);
    if (record === null) {
      throw new HttpException(
        `No user Record with ID '${userId}' Found to Fetch`,
        HttpStatus.NOT_FOUND,
      );
    }
    return record;
  }
}
