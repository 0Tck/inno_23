import { IsNotEmpty } from 'class-validator';

export class CreateUserDto {
  userId?: number;

  @IsNotEmpty()
  username: string;

  @IsNotEmpty()
  userEmail: string;

  @IsNotEmpty()
  password: string;
}

export class LoginUserDto{
  @IsNotEmpty()
  userEmail: string;

  @IsNotEmpty()
  password: string;
}
