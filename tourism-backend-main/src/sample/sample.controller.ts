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

import { Sample } from './sample.model';
import { SampleService } from './sample.service';
import { CreateSampleDto, UpdateSampleDto } from './dto';

@Controller('sample')
@UseInterceptors(FormatResponseInterceptor)
@ApiTags('Sample')
export class SampleController {
  constructor(private sampleService: SampleService) {}

  @Post('create')
  async createSample(@Body() payload: CreateSampleDto): Promise<Sample> {
    return this.sampleService.createSample(payload);
  }

  @Get('get')
  async getAll(): Promise<Sample[]> {
    const records = await this.sampleService.getAll();
    if (records.length === 0) {
      throw new HttpException(
        'No Sample Records Found to Fetch',
        HttpStatus.NOT_FOUND,
      );
    }
    return records;
  }

  @Get('get/:sampleId')
  async getById(@Param('sampleId') sampleId: number): Promise<Sample> {
    const record = await this.sampleService.getById(sampleId);
    if (record === null) {
      throw new HttpException(
        `No Sample Record with ID '${sampleId}' Found to Fetch`,
        HttpStatus.NOT_FOUND,
      );
    }
    return record;
  }

  @Put('update/:sampleId')
  async updateRecord(
    @Param('sampleId') sampleId: number,
    @Body() payload: UpdateSampleDto,
  ): Promise<Sample> {
    const [affectedCount] = await this.sampleService.updateSample(
      sampleId,
      payload,
    );
    if (affectedCount === 0) {
      throw new HttpException(
        `No Sample Record with ID '${sampleId}' Found to Update`,
        HttpStatus.NOT_FOUND,
      );
    } else {
      return this.sampleService.getById(sampleId);
    }
  }

  @Delete('delete/:sampleId')
  async deleteSample(
    @Param('sampleId') sampleId: number,
  ): Promise<{ rowsDeleted: number }> {
    const rowsDeleted = await this.sampleService.deleteSample(sampleId);
    if (rowsDeleted === 0) {
      throw new HttpException(
        `No Sample Record with ID '${sampleId}' Found to Delete`,
        HttpStatus.NOT_FOUND,
      );
    }
    return { rowsDeleted: rowsDeleted };
  }
}
